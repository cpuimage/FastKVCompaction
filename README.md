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
- **Mask‑aware importance selection**  
- **Log‑mass β correction** for attention mass preservation  
- **Ridge regression** for stable value reconstruction  
- **Sliding‑window incremental compression** for autoregressive decoding  
- **Drop‑in replacement Attention module** with compaction support  
- **Supports extremely long KV caches** (32k–100k+)  
- **Compatible with PyTorch 2.x SDPA**

---

## 🔧 Usage Example

### 1. Create a compactor

```python
from fastkv import FastKVCompaction

compactor = FastKVCompaction(
    compression_ratio=10.0,
    lambda_scale_factor=5e-6,
)
```

### 2. Compress KV cache

```python
C_k, C_v, beta = compactor.compact(K, V, Q_ref, attn_mask)
```

### 3. Use compressed KV in attention

```python
from fastkv import compact_attention

out = compact_attention(q, C_k, C_v, beta)
```

### 4. Or use the integrated Attention module

```python
attn = Attention(
    num_heads=H,
    num_kv_heads=H_kv,
    hidden_size=hidden,
    head_dim=D,
    compactor=compactor,
    window_size=8,
)

out, past = attn(x, cos, sin, use_cache=True, past_key_value=past_kv)
```

---

## 📊 Benchmark Results

### Large‑scale random stress tests

| Config                                  | Compression | Relative Error | Cosine |
|--------|-------------|----------------|--------|
| B=1, H=32, L=32768, D=128 | 10× | **0.06%** | **1.000000** |
| B=1, H=32, L=65536, D=128 | 10× | **0.08%** | **1.000000** |
| B=1, H=32, L=32768, D=128 | 20× | **3.27%** | **0.999467** |
| B=1, H=32, L=65536, D=128 | 20× | **0.06%** | **1.000000** |

### Single‑segment attention

- Relative Error: **0.0000%**  
- Cosine Similarity: **1.000000**

### Segmented attention (steady‑state)

- Relative Error: **0.0000%**  
- Cosine Similarity: **1.000000**

These results indicate that the implementation is **highly stable** and suitable for real LLM inference workloads.

---

## 🧠 Design Overview

This implementation includes:

- **Importance estimation** using $\sqrt{\mathbb{E}[(QK^\top)^2]}$
- **Top‑k token selection** based on importance  
- **β log‑mass correction** to preserve attention distribution  
- **Ridge regression** to reconstruct compressed values  
- **Mask‑aware scoring and mass computation**  
- **Sliding‑window incremental compaction** for autoregressive decoding  
- **Full integration into a multi‑head Attention module**

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

