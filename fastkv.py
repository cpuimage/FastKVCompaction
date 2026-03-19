import math
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class FastKVCompaction:
    """
    Fast KV Cache Compaction via Attention Matching (Unofficial Implementation)
    Original Paper: https://arxiv.org/abs/2602.16284v1
    """

    def __init__(
            self,
            compression_ratio=10.0,  # r = L / L_compressed
            lambda_scale_factor=5e-6,  # λ_base: base regularization strength
            min_compression_length=256,  # L_min: minimum compressed length
            sink_size=16,  # n_sink: number of attention sinks to preserve
            logmass_l_chunk=4096,  # L_chunk: chunk size for KV processing
            chunk_R=2048,  # R_chunk: chunk size for reference queries
            lambda_ratio_ref=10.0,  # λ_ref: reference for adaptive regularization
            eps=5e-9  # ε: small constant for numerical stability
    ):
        self.compression_ratio = compression_ratio
        self.lambda_scale_factor = lambda_scale_factor
        self.min_compression_length = min_compression_length
        self.sink_size = sink_size
        self.logmass_l_chunk = logmass_l_chunk
        self.lambda_ratio_ref = lambda_ratio_ref
        self.chunk_R = chunk_R
        self.eps = eps

    @staticmethod
    def _process_mask(attn_mask, batch_size, num_kv_heads, seq_length):
        """
        Unifies various attention mask shapes into a flattened [B*H, L] format.
        Supports shapes: (B, L), (B, 1, L), (B, H, 1, L), or (B, H, R, L).
        """
        if attn_mask is None:
            return None
        mask = attn_mask if attn_mask.dtype == torch.bool else (attn_mask > 0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).expand(batch_size, num_kv_heads, seq_length)
        elif mask.dim() == 3:
            if mask.size(1) == 1:
                mask = mask.expand(batch_size, num_kv_heads, seq_length)
        elif mask.dim() == 4:
            # Assume causal or global mask; take the last query's view
            mask = mask[:, :, -1, :]
        return mask.reshape(batch_size * num_kv_heads, seq_length)

    @staticmethod
    def _online_logsumexp(log_probs_prev: torch.Tensor, log_probs_chunk: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable online Log-Sum-Exp for streaming log-mass accumulation.

        Args:
            log_probs_prev: Previous log probabilities [batch_heads, num_queries]
            log_probs_chunk: Current chunk log probabilities [batch_heads, num_queries, chunk_size]

        Returns:
            Updated log probabilities [batch_heads, num_queries]
        """
        max_prev = log_probs_prev
        max_chunk = log_probs_chunk.max(dim=-1).values
        max_combined = torch.maximum(max_prev, max_chunk)
        max_combined = max_combined.masked_fill(max_combined == float("-inf"), 0.0)

        return max_combined + torch.log(
            torch.exp(log_probs_prev - max_combined) +
            torch.exp(log_probs_chunk - max_combined.unsqueeze(-1)).sum(dim=-1)
        )

    def _solve_ridge(
            self,
            gram_matrix: torch.Tensor,  # G = X^T X ∈ [L_c, L_c]
            projection_matrix: torch.Tensor,  # P = X^T Y ∈ [L_c, D]
            identity: torch.Tensor,  # I ∈ [L_c, L_c]
            regularization: torch.Tensor,  # λ (scalar)
            dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Solve ridge regression problem: (G + λI) V_opt = P

        Args:
            gram_matrix: Gram matrix G = X^T X
            projection_matrix: Projection matrix P = X^T Y
            identity: Identity matrix I
            regularization: Regularization strength λ
            dtype: Output data type

        Returns:
            Optimal value matrix V_opt with same shape as projection_matrix
        """
        # Regularized Gram matrix
        G_reg = gram_matrix + regularization * identity
        # Force symmetry to prevent numerical drift
        G_reg = (G_reg + G_reg.T) * 0.5

        try:
            # Fast path: Cholesky decomposition (requires positive definite matrix)
            L = torch.linalg.cholesky(G_reg)
            V_opt = torch.cholesky_solve(projection_matrix, L).to(dtype)
        except (torch._C._LinAlgError, RuntimeError):
            # Robust path: General linear solver (LU decomposition)
            V_opt = torch.linalg.solve(G_reg, projection_matrix).to(dtype)
        return V_opt

    @torch.no_grad()
    def compact(
            self,
            key_states: torch.Tensor,  # K: [B, H, L, D]
            value_states: torch.Tensor,  # V: [B, H, L, D]
            reference_queries: torch.Tensor,  # Q_ref: [B, H, R, D] or [B, R, D] or [R, D]
            attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV cache via attention matching.

        Returns:
            compressed_keys: K_c [B, H, L_c, D]
            compressed_values: V_c [B, H, L_c, D]
            log_bias: β [B, H, L_c] (additive bias for attention scores)
        """
        batch_size, num_heads, seq_length, head_dim = key_states.shape
        device = key_states.device
        dtype = key_states.dtype
        batch_heads = batch_size * num_heads
        scale = head_dim ** -0.5

        # 1) Determine compressed length L_c ---
        compressed_length = max(self.min_compression_length, int(seq_length / self.compression_ratio))
        if compressed_length >= seq_length:  # Skip if already within budget
            zero_bias = torch.zeros(batch_size, num_heads, seq_length, device=device, dtype=dtype)
            return key_states, value_states, zero_bias

        # 2) Standardize reference queries Q -> [batch_heads, num_queries, head_dim] ---
        queries = reference_queries.float()
        if queries.dim() == 2:  # [R, D]
            queries = queries.view(1, 1, -1, head_dim).expand(batch_size, num_heads, -1, head_dim)
        elif queries.dim() == 3:  # [B, R, D] or [H, R, D]
            if queries.size(0) == batch_size:  # [B, R, D]
                queries = queries.unsqueeze(1).expand(batch_size, num_heads, -1, head_dim)
            else:  # [H, R, D]
                queries = queries.unsqueeze(0).expand(batch_size, num_heads, -1, head_dim)
        # else: [B, H, R, D] - keep as is
        queries = queries.reshape(batch_heads, -1, head_dim)
        num_queries = queries.size(1)  # R

        # 3) Process attention mask ---
        flat_mask = self._process_mask(attention_mask, batch_size, num_heads, seq_length)
        valid_mask = flat_mask if flat_mask is not None else torch.ones(batch_heads, seq_length, device=device,
                                                                        dtype=torch.bool)

        # 4) Importance scoring s_j for each token ---
        # Use a window of recent queries to estimate importance (memory efficient)
        window_size = min(self.sink_size * 2, max(seq_length // 4, 1))
        recent_queries = queries[:, -window_size:, :]

        # Compute attention score magnitudes as importance metric
        attention_scores = torch.bmm(
            recent_queries,
            key_states.reshape(batch_heads, seq_length, head_dim).float().transpose(-1, -2)
        ) * scale

        if flat_mask is not None:
            attention_scores = attention_scores.masked_fill(~valid_mask.unsqueeze(1), 0.0)

        importance_scores = attention_scores.abs().mean(dim=1)  # [batch_heads, seq_length]
        del attention_scores

        # Enforce preservation of attention sinks (initial tokens)
        if self.sink_size > 0:
            importance_scores[:, :self.sink_size] = float("inf")
        importance_scores = importance_scores.masked_fill(~valid_mask, float("-inf"))

        # 5) Top-K selection to form compressed basis ---
        _, top_indices = torch.topk(importance_scores, compressed_length, dim=-1, sorted=False)
        top_indices = top_indices.sort(dim=-1).values  # Maintain chronological order
        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, head_dim)

        # Extract selected keys and values to form K_c and initial V_c
        compressed_keys = torch.gather(
            key_states.reshape(batch_heads, seq_length, head_dim),
            1, gather_indices
        ).float()  # [batch_heads, L_c, head_dim]

        compressed_values_init = torch.gather(
            value_states.reshape(batch_heads, seq_length, head_dim),
            1, gather_indices
        ).float()

        # 6) Streaming log-mass calculation for original attention (log Z_orig) ---
        log_mass_original = torch.full((batch_heads, num_queries), float("-inf"), device=device, dtype=torch.float32)

        for start in range(0, seq_length, self.logmass_l_chunk):
            end = min(seq_length, start + self.logmass_l_chunk)
            chunk_scores = torch.bmm(
                queries,
                key_states.reshape(batch_heads, seq_length, head_dim)[:, start:end, :].float().transpose(-1, -2)
            ) * scale

            if flat_mask is not None:
                chunk_scores = chunk_scores.masked_fill(~valid_mask[:, start:end].unsqueeze(1), float("-inf"))

            log_mass_original = self._online_logsumexp(log_mass_original, chunk_scores)
            del chunk_scores

        # 7) Compressed log-mass and bias β alignment ---
        compressed_scores = torch.bmm(queries, compressed_keys.transpose(-1, -2)) * scale
        log_mass_compressed = torch.logsumexp(compressed_scores, dim=-1)

        # Compute bias β to compensate for mass loss during compression
        log_mass_difference = (log_mass_original - log_mass_compressed).nan_to_num_(0.0)
        beta_per_head = log_mass_difference.median(dim=1, keepdim=True).values  # [batch_heads, 1]

        # 8) Adaptive regularization parameters ---
        severity_factor = max(self.compression_ratio / self.lambda_ratio_ref, 1.0)
        lambda_base = self.lambda_scale_factor / (severity_factor * severity_factor)

        # Pre-allocate output container for compressed values
        compressed_values = torch.empty((batch_heads, compressed_length, head_dim), device=device, dtype=dtype)
        identity_matrix = torch.eye(compressed_length, device=device, dtype=torch.float32)

        # Reshape mask for attention computation if needed
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                current_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            elif attention_mask.dim() == 3:
                current_mask = attention_mask.unsqueeze(2)
            else:
                current_mask = attention_mask
        else:
            current_mask = None

        # --- Memory-efficient head-wise processing ---
        # Process each head independently to avoid storing large Gram matrices for all heads simultaneously
        for head_idx in range(batch_heads):
            # Map flat head index back to batch and head indices
            batch_idx = head_idx // num_heads
            head_idx_in_batch = head_idx % num_heads

            # Initialize Gram matrix G_i and projection matrix P_i for current head
            gram_matrix_i = torch.zeros((compressed_length, compressed_length), device=device, dtype=torch.float32)
            projection_matrix_i = torch.zeros((compressed_length, head_dim), device=device, dtype=torch.float32)

            # Process reference queries in chunks
            for query_start in range(0, num_queries, self.chunk_R):
                query_end = min(num_queries, query_start + self.chunk_R)

                # 1) Compute weight matrix W_chunk for current head and query chunk
                # W = softmax(scores_comp + β) ∈ [R_chunk, L_c]
                scores_chunk = compressed_scores[head_idx:head_idx + 1, query_start:query_end, :].float()
                scores_chunk = scores_chunk + beta_per_head[head_idx].float()
                weight_matrix_chunk = torch.softmax(scores_chunk, dim=-1).squeeze(0)  # [R_chunk, L_c]

                # 2) Incrementally accumulate Gram matrix: G_i += W_chunk^T @ W_chunk
                # Using addmm_ for in-place accumulation to save memory
                gram_matrix_i.addmm_(weight_matrix_chunk.t(), weight_matrix_chunk)

                # 3) Reconstruct original attention output Y_chunk using SDPA
                if current_mask is not None:
                    # Extract mask for current batch, head, and query chunk
                    mask_chunk = current_mask[batch_idx:batch_idx + 1, head_idx_in_batch:head_idx_in_batch + 1,
                    query_start:query_end, :]
                else:
                    mask_chunk = None

                # Compute Y_chunk = Attention(Q_chunk, K, V)
                y_chunk = F.scaled_dot_product_attention(
                    queries[head_idx:head_idx + 1, query_start:query_end, :].view(1, 1, -1, head_dim).to(
                        key_states.dtype),
                    key_states[batch_idx:batch_idx + 1, head_idx_in_batch:head_idx_in_batch + 1, :, :],
                    value_states[batch_idx:batch_idx + 1, head_idx_in_batch:head_idx_in_batch + 1, :, :],
                    attn_mask=mask_chunk
                ).reshape(-1, head_dim)

                # 4) Incrementally accumulate projection matrix: P_i += W_chunk^T @ Y_chunk
                projection_matrix_i.addmm_(weight_matrix_chunk.t(), y_chunk.float())

                # Clean up chunk tensors
                del weight_matrix_chunk, y_chunk, scores_chunk

            # 5) Compute adaptive regularization strength for current head
            gram_trace_i = torch.diagonal(gram_matrix_i).sum()
            regularization_strength = (lambda_base * gram_trace_i / compressed_length) + self.eps

            # 6) Solve ridge regression for current head: (G_i + λI) V_opt_i = P_i
            compressed_values[head_idx] = self._solve_ridge(
                gram_matrix=gram_matrix_i,
                projection_matrix=projection_matrix_i,
                identity=identity_matrix,
                regularization=regularization_strength,
                dtype=dtype
            )

            # Free head-specific tensors
            del gram_matrix_i, projection_matrix_i

        # 9) Package outputs ---
        del compressed_scores

        compressed_keys = compressed_keys.view(batch_size, num_heads, compressed_length, head_dim).to(dtype)
        compressed_values = compressed_values.view(batch_size, num_heads, compressed_length, head_dim)

        # β is applied as additive bias to attention scores in later layers
        log_bias = beta_per_head.expand(batch_heads, compressed_length).reshape(batch_size, num_heads,
                                                                                compressed_length).to(dtype)

        return compressed_keys, compressed_values, log_bias


def compact_attention(q, k, v, beta, attn_mask=None):
    """
    Compute attention using compressed KV cache.

    Args:
        q: [B, H, T, D]
        k: [B, H, L, D]
        v: [B, H, L, D]
        beta: [B, H, Lc] — log-mass correction for compressed prefix
        attn_mask: [B, H, 1, L] or [B, H, T, L]

    Returns:
        [B, H, T, D] — attention output
    """
    B, H, T, D = q.shape
    Lc = beta.size(-1)

    scale = 1.0 / math.sqrt(D)

    # Compute raw attention scores
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Apply beta correction to compressed prefix
    scores[:, :, :, :Lc] += beta.unsqueeze(-2)

    # Apply mask if provided
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class Attention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, hidden_size,
                 head_dim, compactor=None, window_size=0):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.compactor = compactor
        self.window_size = window_size
        self.compaction_threshold = 1024
        self.ref_len = 8
        self.beta_scale = 0.5

    def repeat_kv(self, x):
        return x.repeat_interleave(self.num_key_value_groups, dim=1)

    def _normalize_kv_mask(self, attn_mask, B, H_kv, L, device):
        """
        Normalize mask to shape [B, H_kv, L]
        """
        if attn_mask is None:
            return None

        if attn_mask.dim() == 3:  # [B, H_kv, L] or [B, 1, L]
            if attn_mask.size(1) == 1:
                attn_mask = attn_mask.expand(B, H_kv, L)
            else:
                assert attn_mask.size(1) == H_kv
                assert attn_mask.size(2) == L

        elif attn_mask.dim() == 2:  # [B, L]
            attn_mask = attn_mask.unsqueeze(1).expand(B, H_kv, L)

        elif attn_mask.dim() == 4:  # [B, H_kv/1, T, L] → take T=1
            attn_mask = attn_mask[..., 0, :]
            return self._normalize_kv_mask(attn_mask, B, H_kv, L, device)

        else:
            raise ValueError("Unsupported attn_mask shape")

        return attn_mask.to(device=device)

    def forward(self, x, cos, sin, use_cache=False, past_key_value=None, attn_mask=None):
        B, T, _ = x.shape
        device = x.device

        # Project Q/K/V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = q * cos[:T].unsqueeze(0).unsqueeze(0)
        k = k * cos[:T].unsqueeze(0).unsqueeze(0)

        past_k = past_v = None
        cache_is_compacted = False

        # Load past KV cache
        if past_key_value is not None:
            if len(past_key_value) == 3:
                past_k, past_v, cache_is_compacted = past_key_value
            else:
                past_k, past_v = past_key_value

            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        kv_len = k.size(2)

        # Normalize KV-level mask
        kv_mask = self._normalize_kv_mask(attn_mask, B, self.num_kv_heads, kv_len, device)

        # Determine whether to compact
        use_compaction = cache_is_compacted or (
                self.compactor is not None
                and past_k is not None
                and T == 1
                and kv_len > (self.compaction_threshold + self.window_size)
        )

        beta = None

        if use_compaction:
            split_pos = kv_len - self.window_size

            old_k = k[:, :, :split_pos, :]
            old_v = v[:, :, :split_pos, :]
            new_k = k[:, :, split_pos:, :]
            new_v = v[:, :, split_pos:, :]

            if kv_mask is not None:
                old_mask = kv_mask[:, :, :split_pos]
                new_mask = kv_mask[:, :, split_pos:]
            else:
                old_mask = new_mask = None

            if old_k.size(2) > 0:
                ref_len = min(self.ref_len, q.size(2))
                q_ref = q[:, :, -ref_len:, :].mean(dim=(0, 1))

                C_k, C_v, beta = self.compactor.compact(old_k, old_v, q_ref, attention_mask=old_mask)
                beta = beta * self.beta_scale

                C_k_full = self.repeat_kv(C_k)
                C_v_full = self.repeat_kv(C_v)

                new_k_full = self.repeat_kv(new_k)
                new_v_full = self.repeat_kv(new_v)

                k = torch.cat([C_k_full, new_k_full], dim=2)
                v = torch.cat([C_v_full, new_v_full], dim=2)

                # Build new KV mask (compressed prefix = all True)
                if new_mask is not None:
                    C_mask = torch.ones(
                        B, self.num_kv_heads, C_k.size(2),
                        dtype=torch.bool, device=device
                    )
                    kv_mask = torch.cat([C_mask, new_mask], dim=-1)
                else:
                    kv_mask = torch.ones(
                        B, self.num_kv_heads, C_k.size(2) + new_k.size(2),
                        dtype=torch.bool, device=device
                    )

            else:
                # No old_k → fallback to normal repeat_kv
                k = self.repeat_kv(k)
                v = self.repeat_kv(v)

        else:
            k = self.repeat_kv(k)
            v = self.repeat_kv(v)

        # Build head-level attention mask
        if kv_mask is not None:
            head_mask = kv_mask.repeat_interleave(self.num_key_value_groups, dim=1)
            head_mask = head_mask.unsqueeze(2)
        else:
            head_mask = None

        # Use compact attention if compaction happened
        if use_compaction and beta is not None:
            beta_heads = beta.repeat_interleave(self.num_key_value_groups, dim=1)
            out = compact_attention(q, k, v, beta_heads, attn_mask=head_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=head_mask)

        out = out.transpose(1, 2).reshape(B, T, self.hidden_size)
        out = self.o_proj(out)

        if use_cache:
            if use_compaction:
                past = (k, v, True)
            else:
                past = (k, v)
        else:
            past = None

        return out, past


# ============================================================
# Test: Full Attention Module with FastKVCompaction
# ============================================================

def test_attention_module():
    print("\n=== Testing Attention Module (with FastKVCompaction + attn_mask) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B = 1
    T = 1
    H = 8
    H_kv = 8
    D = 64
    hidden = H * D
    window_size = 8

    compactor = FastKVCompaction(
        compression_ratio=10.0,
    )

    attn = Attention(
        num_heads=H,
        num_kv_heads=H_kv,
        hidden_size=hidden,
        head_dim=D,
        compactor=compactor,
        window_size=window_size
    ).to(device)

    x = torch.randn(B, T, hidden, device=device)
    cos = torch.randn(6000, D, device=device)
    sin = torch.randn(6000, D, device=device)

    past_k = torch.randn(B, H_kv, 6000, D, device=device)
    past_v = torch.randn(B, H_kv, 6000, D, device=device)

    # Mask length must match KV length: 6000 (past) + 1 (current) = 6001
    mask = torch.ones(B, H_kv, 6001, dtype=torch.bool, device=device)
    mask[:, :, 3000:] = False

    attn.eval()
    out, past = attn(
        x, cos, sin,
        use_cache=True,
        past_key_value=(past_k, past_v),
        attn_mask=mask
    )

    print("Output shape:", out.shape)
    print("KV cache length:", past[0].size(2))
    print("Compacted:", len(past) == 3)


# ============================================================
# Test: Single-Segment Attention
# ============================================================

def test_attention_usage():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    H = 8
    L = 2048
    T = 1
    D = 64

    Q = torch.randn(B, H, T, D, device=device)
    Q_ref = Q.reshape(-1, D)

    K = torch.randn(B, H, L, D, device=device)
    V = torch.randn(B, H, L, D, device=device)

    # Randomly mask some tokens
    attn_mask = (torch.rand(B, H, L, device=device) > 0.2)

    out_orig = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask.unsqueeze(2)
    )

    compactor = FastKVCompaction(compression_ratio=20.0)
    C_k, C_v, beta = compactor.compact(K, V, Q_ref, attention_mask=attn_mask)

    compact_mask = torch.ones(
        B, H, C_k.size(2), dtype=torch.bool, device=device
    ).unsqueeze(2)  # [B,H,1,Lc]

    out_compact = compact_attention(Q, C_k, C_v, beta, attn_mask=compact_mask)

    rel_err = (out_compact - out_orig).norm() / out_orig.norm()
    cosine = F.cosine_similarity(out_orig.flatten(), out_compact.flatten(), dim=0)

    print("=== Single-Segment Attention Compression Test ===")
    print(f"Relative Error: {rel_err.item():.4%}")
    print(f"Cosine Similarity: {cosine.item():.6f}")


# ============================================================
# Segmented Attention Test (Steady-State Parameters)
# ============================================================

def test_segmented_compaction():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    H = 8
    L = 2048
    T = 1
    D = 64
    window_size = 8

    Q = torch.randn(B, H, T, D, device=device)
    Q_ref = Q.reshape(-1, D)

    K = torch.randn(B, H, L, D, device=device)
    V = torch.randn(B, H, L, D, device=device)

    # Mask: last 1/4 tokens are masked out
    attn_mask = torch.ones(B, H, L, dtype=torch.bool, device=device)
    attn_mask[:, :, int(0.75 * L):] = False

    out_orig = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask.unsqueeze(2)
    )

    split_pos = L - window_size
    old_K = K[:, :, :split_pos, :]
    old_V = V[:, :, :split_pos, :]
    new_K = K[:, :, split_pos:, :]
    new_V = V[:, :, split_pos:, :]

    old_mask = attn_mask[:, :, :split_pos]
    new_mask = attn_mask[:, :, split_pos:]

    compactor = FastKVCompaction(compression_ratio=20.0)

    C_k, C_v, beta = compactor.compact(old_K, old_V, Q_ref, attention_mask=old_mask)

    # Concatenate compressed old tokens + new tokens
    K_full = torch.cat([C_k, new_K], dim=2)
    V_full = torch.cat([C_v, new_V], dim=2)

    # Mask: compressed part = all True, new part = original mask
    C_mask = torch.ones(B, H, C_k.size(2), dtype=torch.bool, device=device)
    full_mask = torch.cat([C_mask, new_mask], dim=-1).unsqueeze(2)

    out_compact = compact_attention(Q, K_full, V_full, beta, attn_mask=full_mask)

    rel_err = (out_compact - out_orig).norm() / out_orig.norm()
    cosine = F.cosine_similarity(out_orig.flatten(), out_compact.flatten(), dim=0)

    print("=== Segmented Attention Compression Test ===")
    print(f"Relative Error: {rel_err.item():.4%}")
    print(f"Cosine Similarity: {cosine.item():.6f}")


# ============================================================
# FastKVCompaction Full Benchmark
# ============================================================

def test_fastkv_compaction():
    configs = [
        (1, 32, 32768, 128, 10.0),
        (1, 32, 65536, 128, 10.0),
        (1, 32, 32768, 128, 15.0),
        (1, 32, 65536, 128, 15.0),
        (1, 32, 32768, 128, 20.0),
        (1, 32, 65536, 128, 20.0),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for B, H, L, D, ratio in configs:
        print(f"\nConfig: B={B}, H={H}, L={L}, D={D}, Ratio={ratio}x")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        K = torch.randn(B, H, L, D, device=device) * 0.1
        V = torch.randn(B, H, L, D, device=device) * 0.1
        Q_test = torch.randn(B, H, 100, D, device=device) * 0.1
        Q_ref = Q_test[:, :, -8:, :].reshape(-1, D)  # [8*B*H, D]

        # Randomly mask some tokens
        attn_mask = (torch.rand(B, H, L, device=device) > 0.1)

        compactor = FastKVCompaction(compression_ratio=ratio)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        C_k, C_v, beta = compactor.compact(K, V, Q_ref, attention_mask=attn_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        out_orig = F.scaled_dot_product_attention(
            Q_test, K, V,
            attn_mask=attn_mask.unsqueeze(2)
        )

        compact_mask = torch.ones(
            B, H, C_k.size(2), dtype=torch.bool, device=device
        ).unsqueeze(2)

        out_compact = compact_attention(
            Q_test, C_k, C_v, beta,
            attn_mask=compact_mask
        )

        rel_err = (out_compact - out_orig).norm() / out_orig.norm()
        cosine = F.cosine_similarity(out_orig.flatten(), out_compact.flatten(), dim=0)

        print(f"  ✓ Compression: {L} -> {C_k.size(2)} ({L / C_k.size(2):.1f}x)")
        print(f"  ✓ Time: {elapsed * 1000:.1f} ms")
        print(f"  ✓ Relative Error: {rel_err.item():.2%}")
        print(f"  ✓ Cosine Similarity: {cosine.item():.6f}")

        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  ✓ Peak Memory: {mem_gb:.2f} GB")


if __name__ == "__main__":
    print("=" * 70)
    print("Fast KV Cache Compaction")
    print("=" * 70)

    test_fastkv_compaction()
    test_attention_usage()
    test_segmented_compaction()
    test_attention_module()

    print("\n" + "=" * 70)
