# Experiments in Sub-Linear Attention Mechanisms

Single-file PyTorch experiments testing whether a transformer can drop most
attention keys and still win on accuracy — comparing three attention variants
on synthetic event-driven time series:

- `FullAttn` — dense causal, `O(N²)`
- `SlideAttn` — fixed window `w = 20`, `O(N·w)`
- `GatedAttn` — **Weigher Adaptive Attention**: a learned scalar gate per
  position (`g_t = tanh(W_g h_t)`); query `i` attends only to causal keys whose
  gate passes `tau = 0.95`, plus a local band and the diagonal.

Full derivation, mask definition and complexity analysis: [`architecture.md`](architecture.md).

## Files

- [`architecture.md`](architecture.md) — design doc
- [`weigher_adaptive_attention.py`](weigher_adaptive_attention.py) + [`.txt`](weigher_adaptive_attention.txt) run log — next-step **value** prediction (seq len 360)
- [`jepa.py`](jepa.py) + [`.txt`](jepa.txt) run log — same variants under a **JEPA** objective, loss in embedding space (seq len 1000)
- [`requirements.txt`](requirements.txt) — torch, numpy, matplotlib

Shared config: `d_model` 64, batch 32, 5 epochs, Adam `lr 2e-3`, seeds 42.

## Results (from the committed logs)

Value prediction:

```
method      test MSE   ops/query   gate↔event
full          1.1926      360.00        n/a
sliding       1.0644       20.00        n/a
gated         0.5460       26.55      -0.182
```

JEPA:

```
method     JEPA loss   ops/query   gate↔event
full          0.7876     1000.00        n/a
sliding       0.3231       20.00        n/a
gated         0.3212       15.41      -0.043
```

Gated wins the value task outright (best MSE at ~1/14 the arithmetic) and
matches sliding on JEPA loss at slightly lower cost.

## Run

```bash
pip install -r requirements.txt
python weigher_adaptive_attention.py
python jepa.py
```

Gotcha: the figure saves to `../results/weigher_adaptive_ts.png` (one level
**above** the repo) — create that dir or edit the `savefig` path.

## Caveats

- Claims reduced attention *arithmetic*, not wall-clock speed: the prototype
  materializes the dense `N×N` matrix and masks it; `ops/query` counts mask
  entries, and no sparse kernel or timing benchmark exists yet.
- `L = SEQ_LEN` in the reference code, so the local band is non-binding — the
  gate alone drives sparsity.
- Gate↔event correlations are small and negative in both runs: learned-gate
  alignment with the oracle events is **not** established here.