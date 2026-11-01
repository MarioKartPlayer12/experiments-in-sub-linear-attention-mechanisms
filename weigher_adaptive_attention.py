 

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42); np.random.seed(42)

# ============== config ==============
N_SEQ, N_TEST, SEQ_LEN = 3000, 300, 360
BATCH, D_MODEL = 32, 64
WINDOW, THRESHOLD = 20, 0.95
LOCAL_WINDOW, SPIKE_DENSITY = 25, 0.04
N_EPOCHS = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== data ==============
def generate_dataset(n, seq_len):
    vals    = np.zeros((n, seq_len))
    targets = np.zeros((n, seq_len))
    gates_o = np.zeros((n, seq_len))
    events  = np.zeros((n, seq_len))
    t_full  = np.linspace(0, 4*np.pi, seq_len)
    trend   = 0.01 * t_full
    seas    = 2.0*np.sin(t_full) + 0.5*np.sin(3*t_full)
    for b in range(n):
        # AR process
        phi, ns = 0.7, 0.5
        x = np.zeros(seq_len)
        x[0] = seas[0] + np.random.randn()*ns
        for i in range(1, seq_len):
            x[i] = seas[i] + phi*(x[i-1]-seas[i-1]) + ns*np.random.randn()
        # rare level shifts (the "important moments")
        n_shifts   = np.random.randint(3, 7)
        shift_t    = sorted(np.random.choice(seq_len-30, n_shifts, replace=False))
        for st in shift_t:
            x[st:]    += np.random.choice([-1,1]) * np.random.uniform(2,5)
            events[b, st] = 1.0
        vals[b]    = x
        targets[b, :-1] = x[1:]
        # gates: local-bias (recent) + spikes at events + extra scatters
        g = np.zeros(seq_len)
        g[-LOCAL_WINDOW:] = 0.75 + 0.15*np.random.rand(LOCAL_WINDOW)         # local bias
        for st in shift_t:
            g[st] = max(g[st], 0.85 + 0.10*np.random.rand())                  # event spike
        n_extra = max(0, int(SPIKE_DENSITY*seq_len) - len(shift_t))
        if n_extra > 0:
            avail = np.ones(seq_len, dtype=bool); avail[-LOCAL_WINDOW:] = False; avail[shift_t] = False
            idx = np.random.choice(np.where(avail)[0], n_extra, replace=False)
            g[idx] = 0.78 + 0.15*np.random.rand(n_extra)                       # scattered
        g += 0.05*np.random.randn(seq_len)
        gates_o[b] = np.tanh(g)
    return (torch.tensor(vals,    dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(gates_o, dtype=torch.float32),
            torch.tensor(events,  dtype=torch.float32))

print("Generating data...")
train_data = generate_dataset(N_SEQ, SEQ_LEN)
test_data  = generate_dataset(N_TEST, SEQ_LEN)
print(f"  Train: {N_SEQ} sequences x {SEQ_LEN} steps")
print(f"  Test:  {N_TEST} sequences")

# ============== positional encoding ==============
def pos_enc(N, D):
    pe = torch.zeros(N, D); pos = torch.arange(N).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D, 2).float() * -(np.log(10000.0)/D))
    pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
    return pe

# ============== attention modules ==============
class FullAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d)
    def forward(self, h):
        B,N,D = h.shape
        q,k,v = self.qkv(h).chunk(3, -1)
        s = (q @ k.transpose(-2,-1)) / D**0.5
        idx = torch.arange(N, device=h.device)
        s = s.masked_fill( ~(idx[:,None] >= idx[None,:])[None], -1e9 )
        a = F.softmax(s, -1)
        return self.o(a @ v), None, B*N*N

class SlideAttn(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.w = w; self.qkv = nn.Linear(d, 3*d); self.o = nn.Linear(d, d)
    def forward(self, h):
        B, N, D = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        s = q @ k.transpose(-2, -1) / D**0.5          # (B, N, N)
        idx = torch.arange(N, device=h.device)
        causal = idx[:, None] >= idx[None, :]
        win    = (idx[:, None] - idx[None, :]) < self.w
        s = s.masked_fill(~(causal & win), -1e9)
        return self.o(F.softmax(s, -1) @ v), None, B * N * self.w

class GatedAttn(nn.Module):
    """The Weigher Adaptive Attention."""
    def __init__(self, d, thr=0.5, lb=200):
        super().__init__()
        self.thr, self.lb = thr, lb
        self.qkv      = nn.Linear(d, 3*d)
        self.gate_h   = nn.Linear(d, 1)        # learned gate head
        self.o        = nn.Linear(d, d)
    def forward(self, h):
        B,N,D = h.shape
        q,k,v  = self.qkv(h).chunk(3, -1)
        g      = torch.tanh(self.gate_h(h).squeeze(-1))           # (B,N) learned
        idx    = torch.arange(N, device=h.device)
        causal = idx[None,:,None] >= idx[None,None,:]
        lbm    = (idx[None,:,None] - idx[None,None,:]) < self.lb
        gm     = g[:,None,:] >= self.thr
        diag   = torch.eye(N, dtype=torch.bool, device=h.device)[None]
        mask   = (causal & lbm & gm) | diag                       # self always allowed
        s      = (q @ k.transpose(-2,-1)) / D**0.5
        s      = s.masked_fill(~mask, -1e9)
        a      = F.softmax(s, -1)
        out    = self.o(a @ v)
        ops    = int(mask.sum(dim=-1).float().sum().item())      # actual ops executed
        return out, g, ops

class TSModel(nn.Module):
    def __init__(self, attn, d=D_MODEL):
        super().__init__()
        self.in_proj = nn.Linear(1, d)
        self.pos     = pos_enc(SEQ_LEN, d)
        self.attn    = attn
        self.head    = nn.Linear(d, 1)
    def forward(self, x):
        h = self.in_proj(x.unsqueeze(-1)) + self.pos.to(x.device)
        out, g, ops = self.attn(h)
        return self.head(out).squeeze(-1), g, ops

# ============== train + eval ==============
def run(name, attn):
    print(f"\n[{name}]")
    model = TSModel(attn).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
    tr_x, tr_y, _, _ = [t.to(device) for t in train_data]
    nb = N_SEQ // BATCH
    for ep in range(N_EPOCHS):
        perm = torch.randperm(N_SEQ); tot = 0
        for b in range(nb):
            idx = perm[b*BATCH:(b+1)*BATCH]
            pred, _, _ = model(tr_x[idx])
            loss = F.mse_loss(pred[:,:-1], tr_y[idx][:,:-1])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}: loss = {tot/nb:.4f}")
    # eval
    te_x, te_y, _, te_ev = [t.to(device) for t in test_data]
    with torch.no_grad():
        pred, g_pred, ops = model(te_x)
        mse = F.mse_loss(pred[:,:-1], te_y[:,:-1]).item()
        ops_q = ops / (N_TEST * SEQ_LEN)
    gc = None
    if g_pred is not None:
        gc = np.corrcoef(g_pred.abs().mean(0).cpu().numpy(),
                          te_ev.mean(0).cpu().numpy())[0,1]
    print(f"  test MSE = {mse:.4f}   ops/query = {ops_q:.2f}"
          + (f"   gate-event corr = {gc:+.3f}" if gc is not None else ""))
    return dict(mse=mse, ops=ops_q, gate_corr=gc,
                pred=pred[0].cpu().numpy(),
                g_pred=(g_pred[0].cpu().numpy() if g_pred is not None else None))

print("\nTraining models...")
results = {
    'full':    run('full',    FullAttn(D_MODEL)),
    'sliding': run('sliding', SlideAttn(D_MODEL, WINDOW)),
    'gated':   run('gated',   GatedAttn(D_MODEL, THRESHOLD, SEQ_LEN)),
}

# ============== visualizations ==============
fig, axes = plt.subplots(5, 1, figsize=(14, 18))

b_idx  = 0
series = test_data[0][b_idx].numpy()
t      = np.arange(SEQ_LEN)
target = test_data[1][b_idx].numpy()
gates  = test_data[2][b_idx].numpy()
events = test_data[3][b_idx].numpy()
ev_idx = np.where(events > 0)[0]
colors = {'full': 'C0', 'sliding': 'C3', 'gated': 'C2'}

# 1. Time series with level shifts + oracle gates overlay
ax = axes[0]
ax.plot(t, series, 'k-', lw=1, alpha=0.8)
for ei in ev_idx: ax.axvline(ei, color='red', alpha=0.3, lw=1.2)
ax.set_title('① Time series with level shifts (red) — oracle gates overlaid (blue)', fontsize=11)
ax.set_ylabel('value'); ax.grid(alpha=.3)
ax2 = ax.twinx()
ax2.plot(t, gates, 'b-', alpha=0.5, lw=1)
ax2.axhline(THRESHOLD, color='orange', ls='--', alpha=.7)
ax2.fill_between(t[-LOCAL_WINDOW:], -1.1, 1.1, alpha=.1, color='green')
ax2.set_ylabel('gate', color='b'); ax2.tick_params(axis='y', labelcolor='b')

# 2. Learned gates (gated model) vs oracle
ax = axes[1]
ax.plot(t, results['gated']['g_pred'], 'g-', lw=1.4, label='learned gate')
ax.plot(t, gates, 'b--', lw=1, alpha=.4, label='oracle gate')
ax.axhline(THRESHOLD, color='orange', ls='--', alpha=.7, label=f'threshold {THRESHOLD}')
for ei in ev_idx: ax.axvline(ei, color='red', alpha=.15)
ax.set_title(f"② Learned vs oracle gates  (event correlation = {results['gated']['gate_corr']:+.3f})", fontsize=11)
ax.set_ylabel('gate'); ax.legend(loc='upper left'); ax.grid(alpha=.3)

# 3. Predictions
ax = axes[2]
ax.plot(t, target, 'k-', lw=2.2, alpha=.6, label='truth')
for n in ['full','sliding','gated']:
    ax.plot(t, results[n]['pred'], color=colors[n], lw=1.2, alpha=.75,
            label=f"{n}  MSE={results[n]['mse']:.3f}")
for ei in ev_idx: ax.axvline(ei, color='red', alpha=.12)
ax.set_title('③ Predictions vs ground truth', fontsize=11)
ax.set_ylabel('value'); ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=.3)

# 4. Absolute error over time
ax = axes[3]
for n in ['full','sliding','gated']:
    err = np.abs(results[n]['pred'] - target)
    ax.plot(t, err, color=colors[n], lw=1.2, alpha=.8,
            label=f"{n}  MAE={err.mean():.3f}")
for ei in ev_idx: ax.axvline(ei, color='red', alpha=.12)
ax.set_title('④ Absolute error — note spikes near level shifts (red)', fontsize=11)
ax.set_ylabel('|error|'); ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=.3)

# 5. Quality vs compute
ax = axes[4]
methods = ['full','sliding','gated']
mses    = [results[m]['mse'] for m in methods]
ops     = [results[m]['ops'] for m in methods]
xp      = np.arange(len(methods))
b1 = ax.bar(xp-.2, mses, width=.4, color='steelblue', alpha=.85, label='Test MSE')
ax.set_ylabel('Test MSE', color='steelblue'); ax.tick_params(axis='y', labelcolor='steelblue')
ax2 = ax.twinx()
b2 = ax2.bar(xp+.2, ops, width=.4, color='darkorange', alpha=.85, label='Ops/query')
ax2.set_ylabel('Ops per query', color='darkorange'); ax2.tick_params(axis='y', labelcolor='darkorange')
ax.set_xticks(xp); ax.set_xticklabels(methods)
for i,(m,o) in enumerate(zip(mses,ops)):
    ax.text(i-.2, m+.005, f'{m:.3f}', ha='center', fontsize=9)
    ax2.text(i+.2, o+30, f'{o:.0f}', ha='center', fontsize=9)
ax.set_title('⑤ Quality vs compute — gated should match full at far less cost', fontsize=11)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent.parent / 'results' / 'weigher_adaptive_ts.png',
            dpi=110, bbox_inches='tight')
plt.show()

print("\n" + "="*62)
print(f"{'method':<10}{'MSE':>10}{'ops/query':>14}{'gate↔event':>16}")
print("-"*62)
for n in methods:
    gc = f"{results[n]['gate_corr']:+.3f}" if results[n]['gate_corr'] is not None else "  n/a"
    print(f"{n:<10}{results[n]['mse']:>10.4f}{results[n]['ops']:>14.2f}{gc:>16}")