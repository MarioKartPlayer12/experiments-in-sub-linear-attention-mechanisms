# Architecture: Weigher Adaptive Attention

## 1. Motivation

Causal self-attention lets every query interact with every past key, costing
`O(N^2)` time and memory per layer. For long time series this dominates the
budget.

Fixed sliding-window attention reduces the cost to `O(N * w)`, but its receptive
field is static: a rare level shift, anomaly, or regime change older than `w`
steps is permanently invisible, no matter how relevant it is to the current
query. The architecture below keeps the cheap local path while letting the model
select which distant positions are worth attending to.

## 2. Weigher Adaptive Attention

Each token produces a scalar gate from its hidden state:

```
g_t = tanh(W_g h_t)  in (-1, 1)
```

For query position `i`, the admissible key set is

```
A(i) = { j : j <= i, i - j < L, g_j >= tau } U { i }
```

where

- `j <= i` enforces causality,
- `i - j < L` is a local band of width `L`,
- `g_j >= tau` is a key-side importance test with threshold `tau`,
- `{ i }` always admits the diagonal, so no query has an empty row.

Attention is then a softmax restricted to `A(i)`:

```
s_ij = (q_i . k_j) / sqrt(d)                 for j in A(i)
a_ij = exp(s_ij) / sum_{k in A(i)} exp(s_ik)
out_i = W_o sum_{j in A(i)} a_ij v_j
```

Design notes:

- **Key-side gating.** The gate is attached to the key position, not the query.
  It decides whether a position is retrievable at all. This is what makes the
  mask indexable: the set of "important" positions is shared across all future
  queries.
- **Local band as a floor.** Recent context is always available even when the
  gates are low, which keeps short-range dynamics (AR terms, seasonality) intact.
- **Diagonal guarantee.** Self-attention is always allowed, preventing empty
  attention rows and giving the predictor a residual-like fallback.
- **Bounded gate.** `tanh` keeps `g_t` in `(-1, 1)`; `tau` directly controls the
  density of admitted keys.

## 3. Complexity

Let `rho(tau) = P(g_j >= tau)` be the gate density and `L` the band width. The
expected number of admissible keys per query is

```
|A(i)| <= L * rho + 1
```

| Variant        | Keys per query | Total cost       |
|----------------|----------------|------------------|
| Full causal    | `N`            | `O(N^2)`         |
| Sliding window | `w`            | `O(N * w)`       |
| Weigher        | `L * rho + 1`  | `O(N * (L*rho + 1))` |

When the gates are sparse (`rho` small), the cost is linear in `N` and
sub-linear relative to full attention. With an index over positions whose gate
passes `tau` (bucketed or binary-searched), each query retrieves its
`O(L * rho)` candidates without scanning the whole history, giving sub-linear
per-query retrieval.

The empirical metric used throughout is

```
ops/query = (1 / N) * sum_i |A(i)|
```

computed from the realized mask. It upper-bounds the number of attention dot
products and excludes index lookup overhead.

Two implementation caveats matter for interpreting the numbers:

- In the reference implementation `L = SEQ_LEN`, so the band is non-binding and
  the gate alone determines sparsity. Choosing a smaller `L` activates the band
  and bounds worst-case cost even when many gates pass.
- The prototype materializes the dense `N x N` score matrix and masks it.
  `ops/query` therefore measures the arithmetic that a sparse kernel would need,
  not the runtime of this dense prototype. A production kernel would use
  block-sparse or gather-based attention.

## 4. Reference Implementation

The modules live in `experiments/weigher_adaptive_attention.py`:

- `FullAttn` — dense causal baseline, `O(N^2)`.
- `SlideAttn` — fixed window `w`, `O(N * w)`.
- `GatedAttn` — the Weigher Adaptive Attention described above. The gate head is
  a single linear layer followed by `tanh`; the mask is
  `(causal & band & gate) | diagonal`; `ops` is the number of mask entries that
  are `True`.

## 5. JEPA Variant

Next-token prediction trains the model to reconstruct the next value. JEPA
instead predicts the next *embedding*:

```
h      = Encoder(x) + positional_encoding
u      = Attention(h)
p_i    = Predictor(u_i)                 # predicts h_{i+1}
target = stopgrad(roll(h, -1))
L_JEPA = MSE(p_i, target_i)             # for i < N - 1
```

An auxiliary value-space decoder is trained alongside it:

```
L_aux = MSE(Decoder(p_i), x_{i+1})
L     = L_JEPA + 0.1 * L_aux
```

The decoder is used only for visualization and does not change the
representation objective.

The attention module is identical in both variants, so the complexity analysis
above carries over unchanged. The stop-gradient on the target is the standard
JEPA safeguard against representation collapse through a shortcut in the
predictor. This is a simplified JEPA setup: the target is the stop-gradient
online embedding rather than an EMA target encoder.

## 6. Scope

The architecture claims a reduction in required attention arithmetic on
event-driven sequences, not superior accuracy on all tasks. The reference code
is a correctness-oriented prototype: dense masking, fixed threshold, no fused
kernel, and no wall-clock benchmark.
