# Experiments in Sub-Linear Attention Mechanisms

Prototype implementations and results for **Weigher Adaptive Attention** — a learned,
input-dependent sparse causal attention — compared against full and sliding-window
attention on synthetic event-driven time series. Includes a JEPA
(next-embedding prediction) variant.

See [docs/architecture.md](docs/architecture.md) for the theoretical architecture.

## Results

Setup for both runs: 3000 train / 300 test sequences, `d_model = 64`, batch size 32,
5 epochs, Adam at `lr = 2e-3`, seed 42, gate threshold `tau = 0.95`, sliding window 20.

### Value prediction (`SEQ_LEN = 360`)

| Method  | Test MSE | ops/query | gate-event corr |
|---------|---------:|----------:|----------------:|
| full    |   1.1926 |    360.00 |             n/a |
| sliding |   1.0644 |     20.00 |             n/a |
| gated   | **0.5460** |  26.55 |          -0.182 |

### JEPA / next-embedding prediction (`SEQ_LEN = 1000`)

| Method  | Test JEPA loss | ops/query | gate-event corr |
|---------|---------------:|----------:|----------------:|
| full    |         0.7876 |   1000.00 |             n/a |
| sliding |         0.3231 |     20.00 |             n/a |
| gated   |     **0.3212** |     15.41 |          -0.043 |

### Notes

- The gated model reaches lower error than full and sliding attention at 26.55 and
  15.41 ops/query, versus 360 and 1000 for full attention.
- The gate-event correlation is negative in both runs: the learned gate does not
  reproduce the synthetic oracle gates. The gain comes from adaptive sparsity, not
  from event detection. This needs further investigation.
- `ops/query` is counted from the realized attention mask. The prototype still builds
  a dense `N x N` score matrix, so wall-clock time is not yet sub-quadratic.

## Layout

```
experiments/
  weigher_adaptive_attention.py   value-prediction comparison
  jepa.py                         next-embedding prediction comparison
results/
  weigher_adaptive_attention.txt  console output
  jepa.txt                        console output
  weigher_adaptive_ts.png         5-panel figure
docs/
  architecture.md                 theory and complexity analysis
```

## Usage

```bash
pip install -r requirements.txt
python experiments/weigher_adaptive_attention.py
python experiments/jepa.py
```

The first script writes its figure to `results/weigher_adaptive_ts.png`.
