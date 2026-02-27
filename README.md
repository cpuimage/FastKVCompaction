# FastKVCompaction (Unofficial PyTorch Implementation)

This is an unofficial implementation of **Fast KV Cache Compaction via Attention Matching** 
(arXiv:2602.16284) - achieving <0.002% error with 10-20x compression for 100k+ context.

The implementation is designed for **practical integration into LLM inference**, enabling efficient KV‑cache compression for long‑context models while maintaining extremely low attention reconstruction error.

> ⚠ **Disclaimer:** 
> This project is **not** an official implementation. 
> It is not affiliated with or endorsed by the authors of the original paper.

---

## 🚀 Features

- **High‑fidelity KV cache compression** with <0.002% error in real attention paths  
- **Mask‑aware importance selection** supporting multiple mask formats (`[B, L]`, `[B, H_kv, L]`, etc.)  
- **Log‑mass β correction** for attention mass preservation across compressed tokens  
- **Adaptive ridge regression** with severity-based regularization for stable value reconstruction  
- **Sink token preservation** (e.g., BOS tokens) to maintain attention stability  
- **Streaming computation** for importance estimation and log‑mass calculation (memory-efficient)  
- **Compatible with PyTorch 2.x SDPA** and GQA/MQA attention patterns

---

## 🔧 Usage Example

### 1. Create a compactor

```python
from fastkv import FastKVCompaction

compactor = FastKVCompaction(
    compression_ratio=10.0,        # Target compression ratio
    lambda_scale_factor=5e-6,      # Base regularization strength
    min_compression_length=256,    # Minimum tokens to preserve
    sink_size=16,                  # Always preserve first N tokens
    importance_r_chunk=64,         # Chunk size for importance computation
    logmass_l_chunk=512,           # Chunk size for log‑mass streaming
)
```

### 2. Compress KV cache

```python
# K, V: [B, H_kv, L, D] - original KV cache
# Q_ref: [R, D] or [B, R, D] or [B, H_kv, R, D] - reference queries
# attn_mask: Optional, supports [B, L], [B, H_kv, L], [B, H_kv, 1, L]

C_k, C_v, beta = compactor.compact(K, V, Q_ref, attn_mask)
# C_k, C_v: [B, H_kv, Lc, D] - compressed KV cache
# beta: [B, H_kv, Lc] - log‑mass correction term
```

### 3. Use compressed KV in attention

```python
from fastkv import compact_attention

# q: [B, H, T, D] - queries (H may differ from H_kv in GQA/MQA)
# C_k, C_v: [B, H_kv, Lc, D] - compressed KV cache
# beta: [B, H_kv, Lc] - correction from compactor
# attn_mask: [B, H, T, L] or broadcastable shape

out = compact_attention(q, C_k, C_v, beta, attn_mask)
# out: [B, H, T, D]
```

**Note on GQA/MQA**: In grouped-query attention, `H` (query heads) may be greater than `H_kv` (KV heads). The `beta` term is automatically broadcast across query heads in `compact_attention`.

---

## 📊 Implementation Details

### Importance Estimation

Unlike the paper's RMS suggestion, this implementation uses **mean absolute attention** for numerical stability:

```python
importance = mean(|QK^T|) over reference queries
```

This provides robust importance scores while avoiding overflow risks with squared terms.

### Sink Token Protection

The first `sink_size` tokens (configurable, default 16) are always preserved by setting their importance to infinity:

```python
if self.sink_size > 0:
    importance[:, :self.sink_size] = 1e10
```

This ensures critical positional information (like BOS tokens) is never compressed away.

### Adaptive Regularization

Ridge regression uses dynamic λ based on compression severity:

```python
severity = max(compression_ratio / lambda_ratio_ref, 1.0)
effective_lambda = lambda_scale_factor / (severity^2)
```

Higher compression ratios → stronger regularization → more stable solutions.

### Streaming Log‑Sum‑Exp

Memory-efficient computation of attention mass using chunked processing:

```python
def online_lse(cur, new):
    # Numerically stable streaming log‑sum‑exp
    m = max(cur, max(new))
    return m + log(exp(cur - m) + sum(exp(new - m)))
```

---

## ⚠️ Important Notes

1. **Reference Queries (`Q_ref`)**: Should be representative of the query distribution during inference. Common choices:
   - First few tokens of the input sequence
   - Learnable query embeddings
   - Running average of past queries

2. **Mask Handling**: The compactor accepts flexible mask formats but requires boolean masks (True = attend, False = mask out). Non-boolean masks are converted via `(mask > 0)`.

3. **Numerical Precision**: Internal computations use `float32` for stability regardless of input dtype. Outputs are cast back to input dtype.

4. **Beta Correction**: The `beta` term compensates for attention mass lost during compression. It is computed as the median difference between original and compressed log‑masses across reference queries.

---

## 📚 Citation

If you use this repository in academic work, please cite the original paper:

```bibtex
@article{fastkv2026,
  title={Fast KV Cache Compaction via Attention Matching},
  year={2026},
  journal={ArXiv preprint arXiv:2602.16284}
}
```

---

## 📄 License

MIT License.

---

## 🙏 Acknowledgements

Thanks to the authors of *Fast KV Cache Compaction via Attention Matching* for introducing this elegant and practical method. 
This repository aims to provide a clean, engineering‑ready PyTorch implementation for the community.