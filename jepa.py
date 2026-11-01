 
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

torch.manual_seed(42); np.random.seed(42)

# ============ config ============
N_SEQ, N_TEST, SEQ_LEN = 3000, 300, 1000
BATCH, D_MODEL = 32, 64
WINDOW, THRESHOLD = 20, 0.95
LOCAL_WINDOW, SPIKE_DENSITY = 25, 0.04
N_EPOCHS = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============ data ============
def generate_dataset(n, seq_len):
    vals=np.zeros((n,seq_len)); targets=np.zeros((n,seq_len))
    gates_o=np.zeros((n,seq_len)); events=np.zeros((n,seq_len))
    t_full=np.linspace(0,4*np.pi,seq_len); trend=0.01*t_full
    seas=2.0*np.sin(t_full)+0.5*np.sin(3*t_full)
    for b in range(n):
        phi,ns=0.7,0.5; x=np.zeros(seq_len)
        x[0]=seas[0]+np.random.randn()*ns
        for i in range(1,seq_len):
            x[i]=seas[i]+phi*(x[i-1]-seas[i-1])+ns*np.random.randn()
        n_shifts=np.random.randint(3,7)
        shift_t=sorted(np.random.choice(seq_len-30,n_shifts,replace=False))
        for st in shift_t:
            x[st:]+=np.random.choice([-1,1])*np.random.uniform(2,5)
            events[b,st]=1.0
        vals[b]=x; targets[b,:-1]=x[1:]
        g=np.zeros(seq_len); g[-LOCAL_WINDOW:]=0.75+0.15*np.random.rand(LOCAL_WINDOW)
        for st in shift_t: g[st]=max(g[st],0.85+0.10*np.random.rand())
        n_extra=max(0,int(SPIKE_DENSITY*seq_len)-len(shift_t))
        if n_extra>0:
            avail=np.ones(seq_len,dtype=bool); avail[-LOCAL_WINDOW:]=False; avail[shift_t]=False
            ix=np.random.choice(np.where(avail)[0],n_extra,replace=False)
            g[ix]=0.78+0.15*np.random.rand(n_extra)
        g+=0.05*np.random.randn(seq_len); gates_o[b]=np.tanh(g)
    return (torch.tensor(vals,dtype=torch.float32), torch.tensor(targets,dtype=torch.float32),
            torch.tensor(gates_o,dtype=torch.float32), torch.tensor(events,dtype=torch.float32))

print("Generating data...")
train_data = generate_dataset(N_SEQ, SEQ_LEN)
test_data  = generate_dataset(N_TEST, SEQ_LEN)

# ============ positional encoding ============
def pos_enc(N, D):
    pe=torch.zeros(N,D); pos=torch.arange(N).unsqueeze(1).float()
    div=torch.exp(torch.arange(0,D,2).float()*-(np.log(10000.0)/D))
    pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
    return pe

# ============ attention modules ============
class FullAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv=nn.Linear(d,3*d); self.o=nn.Linear(d,d)
    def forward(self, h):
        B,N,D=h.shape; q,k,v=self.qkv(h).chunk(3,-1)
        s=(q@k.transpose(-2,-1))/D**0.5
        idx=torch.arange(N,device=h.device)
        s=s.masked_fill(~(idx[:,None]>=idx[None,:])[None],-1e9)
        return self.o(F.softmax(s,-1)@v), None, B*N*N
class SlideAttn(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.w, self.qkv, self.o = w, nn.Linear(d, 3*d), nn.Linear(d, d)
    def forward(self, h):
        B, N, D = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        idx = torch.arange(N, device=h.device)
        causal    = idx[None, :] <= idx[:, None]                       # (N, N)
        in_window = (idx[None, :] - idx[:, None] + self.w) > 0         # within w steps back
        mask = causal & in_window                                      # (N, N)
        s = (q @ k.transpose(-2, -1)) / D**0.5
        s = s.masked_fill(~mask[None], -1e9)
        return self.o(F.softmax(s, -1) @ v), None, B * N * self.w 


class GatedAttn(nn.Module):
    def __init__(self, d, thr=0.80, lb=200):
        super().__init__()
        self.thr,self.lb=thr,lb
        self.qkv=nn.Linear(d,3*d); self.gate_h=nn.Linear(d,1); self.o=nn.Linear(d,d)
    def forward(self, h):
        B,N,D=h.shape; q,k,v=self.qkv(h).chunk(3,-1)
        g=torch.tanh(self.gate_h(h).squeeze(-1))
        idx=torch.arange(N,device=h.device)
        causal=idx[None,:,None]>=idx[None,None,:]
        lbm=(idx[None,:,None]-idx[None,None,:])<self.lb
        gm=g[:,None,:]>=self.thr
        diag=torch.eye(N,dtype=torch.bool,device=h.device)[None]
        mask=(causal&lbm&gm)|diag
        s=(q@k.transpose(-2,-1))/D**0.5; s=s.masked_fill(~mask,-1e9)
        a=F.softmax(s,-1); out=self.o(a@v)
        ops=int(mask.sum(dim=-1).float().sum().item())
        return out, g, ops

# ============ JEPA ============
class TSModelJEPA(nn.Module):
    """
    JEPA: encoder -> attention -> predictor -> predicted next embedding.
    Loss is in embedding space, not value space.
    A small decoder head is kept for value-space visualization only.
    """
    def __init__(self, attn, d=D_MODEL):
        super().__init__()
        self.encoder   = nn.Linear(1, d)
        self.pos       = pos_enc(SEQ_LEN, d)
        self.attn      = attn                # <<< the only thing we vary
        self.predictor = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))
        self.decoder   = nn.Linear(d, 1)     # auxiliary, for visualization
    def forward(self, x):
        h = self.encoder(x.unsqueeze(-1)) + self.pos.to(x.device)      # (B,N,d)
        out, g, ops = self.attn(h)                                      # attention on h
        pred_h   = self.predictor(out)                                  # (B,N,d)
        # target: next embedding (stop-gradient)
        with torch.no_grad():
            target_h = torch.roll(h, -1, dims=1)
        # decoder: for visualization only
        pred_val = self.decoder(pred_h).squeeze(-1)
        return pred_h[:, :-1], target_h[:, :-1].detach(), g, ops, pred_val

# ============ train + eval (LOSS = JEPA) ============
def run(name, attn):
    print(f"\n[{name}]")
    model = TSModelJEPA(attn).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
    tr_x  = train_data[0].to(device)
    nb    = N_SEQ // BATCH
    for ep in range(N_EPOCHS):
        perm=torch.randperm(N_SEQ); tot=0
        for b in range(nb):
            ix=perm[b*BATCH:(b+1)*BATCH]; x=tr_x[ix]
            pred_h, target_h, g, ops, pred_val = model(x)
            jepa_loss = F.mse_loss(pred_h, target_h)                         # <<< THE loss
            aux_loss  = F.mse_loss(pred_val[:, :-1], x[:, 1:]) * 0.1         # visualization only
            loss = jepa_loss + aux_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}: loss = {tot/nb:.4f}")
    # eval
    te_x, _, _, te_ev = [t.to(device) for t in test_data]
    with torch.no_grad():
        pred_h, target_h, g, ops, pred_val = model(te_x)
        jepa_mse = F.mse_loss(pred_h, target_h).item()
        ops_q = ops / (N_TEST * SEQ_LEN)
    gc = None
    if g is not None:
        gc = np.corrcoef(g.abs().mean(0).cpu().numpy(), te_ev.mean(0).cpu().numpy())[0,1]
    print(f"  test JEPA loss = {jepa_mse:.4f}   ops/query = {ops_q:.2f}"
          + (f"   gate-event corr = {gc:+.3f}" if gc is not None else ""))
    return dict(mse=jepa_mse, ops=ops_q, gate_corr=gc,
                pred=pred_val[0].cpu().numpy(),
                g_pred=(g[0].cpu().numpy() if g is not None else None),
                target=test_data[1][0].numpy())

print("\nTraining JEPA models...")
results = {
    'full':    run('full',    FullAttn(D_MODEL)),
    'sliding': run('sliding', SlideAttn(D_MODEL, WINDOW)),
    'gated':   run('gated',   GatedAttn(D_MODEL, THRESHOLD, SEQ_LEN)),
} 