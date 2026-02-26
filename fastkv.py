import time
import torch
import torch.nn.functional as F
import math

from torch import nn


class FastKVCompaction:
    """
    Fast KV Cache Compaction via Attention Matching (Unofficial Implementation)
    https://arxiv.org/abs/2602.16284v1
    """

    def __init__(
            self,
            compression_ratio: float = 10.0,
            lambda_scale_factor: float = 5e-6,
            min_compression_length: int = 256,
            importance_r_chunk: int = 64,
            logmass_l_chunk: int = 512,
            sink_size: int = 16,
            residual_size: int = 128,
            lambda_ratio_ref: float = 10.0,
    ):
        # Controls how aggressively we compress the KV cache.
        self.compression_ratio = compression_ratio

        # Base scale for ridge regression regularization.
        # Smaller λ → less shrinkage → lower bias but more numerical risk.
        self.lambda_scale_factor = lambda_scale_factor

        # Minimum number of tokens to keep regardless of ratio.
        self.min_compression_length = min_compression_length

        # Chunk sizes for computing importance and log‑mass in streaming fashion.
        self.importance_r_chunk = importance_r_chunk
        self.logmass_l_chunk = logmass_l_chunk

        # Sink tokens are always preserved (e.g., BOS tokens).
        self.sink_size = sink_size

        # Reserved for potential tail‑preservation logic (not used here).
        self.residual_size = residual_size

        # Controls how λ decays with compression severity.
        self.lambda_ratio_ref = lambda_ratio_ref

    @staticmethod
    def _process_mask(attn_mask: torch.Tensor, B: int, H_kv: int, L: int):
        """
        Normalize attention mask into shape [BH, L].
        Supports:
            [B, L]
            [B, 1, L]
            [B, H_kv, L]
            [B, H_kv, 1, L]
        """
        if attn_mask is None:
            return None

        # Convert non‑boolean masks to boolean.
        if attn_mask.dtype != torch.bool:
            mask = (attn_mask > 0)
        else:
            mask = attn_mask

        # Expand to [B, H_kv, L]
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).expand(-1, H_kv, -1)
        elif mask.dim() == 3:
            if mask.size(1) == 1:
                mask = mask.expand(B, H_kv, -1)
        elif mask.dim() == 4:
            mask = mask[:, :, -1, :]  # last query position

        return mask.reshape(B * H_kv, L)

    @torch.no_grad()
    def compact(self, K, V, Q_ref, attn_mask=None):
        """
        Main compaction routine.

        Inputs:
            K, V: [B, H_kv, L, D] — original KV cache
            Q_ref: reference queries used to measure importance
            attn_mask: optional boolean mask

        Outputs:
            C_k: compressed keys  [B, H_kv, Lc, D]
            C_v: compressed values [B, H_kv, Lc, D]
            beta: log‑mass correction [B, H_kv, Lc]
        """
        B, H_kv, L, D = K.shape
        device = K.device
        dtype = K.dtype
        scale = D ** -0.5
        BH = B * H_kv

        # ------------------------------------------------------------
        # 1) Determine compression budget
        # ------------------------------------------------------------
        budget = max(self.min_compression_length, int(L / self.compression_ratio))
        if budget >= L:
            # No compression needed
            beta = torch.zeros(B, H_kv, L, device=device, dtype=dtype)
            return K, V, beta

        # ------------------------------------------------------------
        # 2) Normalize Q_ref → [BH, R, D]
        # ------------------------------------------------------------
        if Q_ref.dim() == 2:
            R = Q_ref.size(0)
            Q_flat = Q_ref.float().view(1, 1, R, D).expand(B, H_kv, R, D).reshape(BH, R, D)
        elif Q_ref.dim() == 3:
            R = Q_ref.size(1)
            Q_flat = Q_ref.float().unsqueeze(1).expand(B, H_kv, R, D).reshape(BH, R, D)
        else:
            R = Q_ref.size(2)
            Q_flat = Q_ref.float().reshape(BH, R, D)

        # Convert K/V to float32 for stable math
        K_f32 = K.float()
        V_f32 = V.float()
        K_flat = K_f32.reshape(BH, L, D)

        # Process mask → [BH, L]
        mask_flat = self._process_mask(attn_mask, B, H_kv, L)
        valid_mask = torch.ones(BH, L, device=device, dtype=torch.bool) if mask_flat is None else mask_flat

        # ------------------------------------------------------------
        # 3) Importance estimation
        #    importance = mean(|QK|) over reference queries
        # ------------------------------------------------------------
        importance = torch.zeros(BH, L, device=device, dtype=torch.float32)
        r_chunk = self.importance_r_chunk or R
        K_t = K_flat.transpose(-1, -2)

        for r_start in range(0, R, r_chunk):
            r_end = min(R, r_start + r_chunk)
            Qc = Q_flat[:, r_start:r_end, :]
            scores = torch.bmm(Qc, K_t) * scale  # [BH, r_chunk, L]

            if mask_flat is not None:
                scores = scores.masked_fill(~valid_mask.unsqueeze(1), 0.0)

            # Stable importance metric
            importance += torch.mean(torch.abs(scores), dim=1)

        # Always keep sink tokens
        if self.sink_size > 0:
            importance[:, :self.sink_size] = 1e10

        importance = importance.masked_fill(~valid_mask, float("-inf"))

        # ------------------------------------------------------------
        # 4) Top‑K selection
        # ------------------------------------------------------------
        _, topk_idx = torch.topk(importance, budget, dim=-1, sorted=False)
        batch_idx = torch.arange(BH, device=device).unsqueeze(-1)
        C_k = K_flat[batch_idx, topk_idx]  # [BH, Lc, D]

        # ------------------------------------------------------------
        # 5) Compute original log‑mass via streaming log‑sum‑exp
        # ------------------------------------------------------------
        def online_lse(cur, new):
            """
            Numerically stable streaming log‑sum‑exp:
                cur = logsumexp(previous_chunks)
                new = scores for current chunk
            """
            m1 = cur
            m2, _ = torch.max(new, dim=-1)
            m = torch.maximum(m1, m2)
            return m + torch.log(
                torch.exp(m1 - m) +
                torch.sum(torch.exp(new - m.unsqueeze(-1)), dim=-1)
            )

        log_mass_orig = torch.full((BH, R), float("-inf"), device=device)
        l_chunk = self.logmass_l_chunk or L

        for l_start in range(0, L, l_chunk):
            l_end = min(L, l_start + l_chunk)
            K_l = K_flat[:, l_start:l_end, :]
            scores_l = torch.bmm(Q_flat, K_l.transpose(-1, -2)) * scale

            if mask_flat is not None:
                scores_l = scores_l.masked_fill(
                    ~valid_mask[:, l_start:l_end].unsqueeze(1),
                    float("-inf"),
                )

            log_mass_orig = online_lse(log_mass_orig, scores_l)
        if mask_flat is not None:
            sdpa_mask = valid_mask.view(B, H_kv, L).unsqueeze(2).expand(-1, -1, R, -1)
        else:
            sdpa_mask = None

        Y_full = F.scaled_dot_product_attention(
            Q_flat.view(B, H_kv, R, D),
            K_f32,
            V_f32,
            attn_mask=sdpa_mask,
        ).reshape(BH, R, D)

        # ------------------------------------------------------------
        # 6) Compute compressed log‑mass
        # ------------------------------------------------------------
        log_mass_comp = torch.empty(BH, R, device=device)
        for i in range(BH):
            Qi = Q_flat[i]
            Cki = C_k[i]
            scores_ci = (Qi @ Cki.transpose(-1, -2)) * scale
            log_mass_comp[i] = torch.logsumexp(scores_ci, dim=-1)

        # β = median(log_mass_orig − log_mass_comp)
        diff = log_mass_orig - log_mass_comp
        beta_per_head = diff.median(dim=1, keepdim=True).values
        beta = beta_per_head.expand(-1, budget).view(B, H_kv, budget).to(dtype)

        # ------------------------------------------------------------
        # 7) Ridge regression to fit compressed values C_v
        # ------------------------------------------------------------
        C_v = torch.empty(BH, budget, D, device=device)

        # Compression severity controls λ decay
        severity = max(self.compression_ratio / self.lambda_ratio_ref, 1.0)
        effective_lambda = self.lambda_scale_factor / (severity * severity)

        I = None

        for i in range(BH):
            Qi = Q_flat[i]
            Cki = C_k[i]

            # Attention weights under compressed keys
            s_i = (Qi @ Cki.transpose(-1, -2)) * scale
            b_i = beta_per_head[i]
            W = torch.softmax(s_i + b_i.unsqueeze(0), dim=-1)

            Xt = W.transpose(-1, -2)
            XtX = Xt @ W
            XtY = Xt @ Y_full[i]

            if I is None or I.size(0) != budget:
                I = torch.eye(budget, device=device)

            trace = torch.diagonal(XtX).sum()
            lam = effective_lambda * trace / max(budget, 1) + 1e-9

            # Solve (XtX + λI) C_v = XtY
            C_v[i] = torch.linalg.solve(XtX + lam * I, XtY)

        # Reshape back to [B, H_kv, Lc, D]
        C_k = C_k.view(B, H_kv, budget, D).to(dtype)
        C_v = C_v.view(B, H_kv, budget, D).to(dtype)

        return C_k, C_v, beta


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

                C_k, C_v, beta = self.compactor.compact(old_k, old_v, q_ref, attn_mask=old_mask)
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
    C_k, C_v, beta = compactor.compact(K, V, Q_ref, attn_mask=attn_mask)

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

    C_k, C_v, beta = compactor.compact(old_K, old_V, Q_ref, attn_mask=old_mask)

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

        C_k, C_v, beta = compactor.compact(K, V, Q_ref, attn_mask=attn_mask)

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
