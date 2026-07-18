import torch, math, statistics
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

# ----- Scaled streamed continual-learning falsification of Spectral Morphogenesis -----
# Sheaf Laplacian L(t) evolves as tasks stream in. Readout = spectral projector P_k onto
# bottom-k eigenspace, then a fixed Lipschitz head h. Forgetting on OLD data is bounded by
# Lip(h)*||x||*||P_perp B P_old|| / gamma_eff  (verified theorem).

d = 128        # node/feature dimension (sheaf stalk dim, scaled up)
k = 8          # bottom-k eigenspace read out
n_eval = 512   # held-out evaluation samples from OLD task distribution
n_tasks = 6    # sequential tasks streamed

def sym(A):
    return 0.5*(A + A.T)

def spectral_readout(L, k):
    # eigh returns ascending eigenvalues; bottom-k eigenvectors span the readout subspace
    w, V = torch.linalg.eigh(sym(L))
    Vk = V[:, :k]
    P = Vk @ Vk.T          # spectral projector onto bottom-k eigenspace
    gap = (w[k] - w[k-1]).item()   # spectral gap gamma
    return P, Vk, w, gap

# Fixed Lipschitz head: linear map W with controlled spectral norm Lip(h)=||W||_2
Wh = torch.randn(4, d, dtype=dtype, device=dev)
Wh = Wh / torch.linalg.matrix_norm(Wh, ord=2)   # normalize so Lip(h) = 1 exactly
Lip_h = torch.linalg.matrix_norm(Wh, ord=2).item()
def head(z):
    return z @ Wh.T

# Base sheaf Laplacian: SPD with a clean bottom-k gap (block structure)
A = torch.randn(d, d, dtype=dtype, device=dev)
L0 = sym(A @ A.T) / d
w0, V0 = torch.linalg.eigh(L0)
# widen the bottom-k gap so the readout subspace is well-defined
w0_mod = w0.clone()
w0_mod[:k] = w0_mod[:k] - 1.5
L0 = sym(V0 @ torch.diag(w0_mod) @ V0.T)

# Held-out OLD-task evaluation data (fixed across the stream), unit-ish norm
Xeval = torch.randn(n_eval, d, dtype=dtype, device=dev)
Xeval = Xeval / Xeval.norm(dim=1, keepdim=True)   # ||x||=1 per sample

def make_coupling(P_old, beta, seed):
    # cochain that streams a new task: symmetric perturbation with a genuine
    # old<->complement coupling block; scaled by beta.
    g = torch.Generator(device=dev); g.manual_seed(seed)
    R = torch.randn(d, d, dtype=dtype, device=dev, generator=g)
    B = beta * sym(R) / math.sqrt(d)
    return B

def run_stream(beta, do_proj):
    L = L0.clone()
    P_old, Vk_old, w, gap = spectral_readout(L, k)
    Iu = torch.eye(d, dtype=dtype, device=dev)
    # baseline readout on OLD data (reference for forgetting)
    pred_ref = head((P_old @ Xeval.T).T)
    max_ratio = 0.0; worst_forget = 0.0
    for t in range(n_tasks):
        B = make_coupling(P_old, beta, 100+t)
        Pperp = Iu - P_old
        # sharp coupling block that drives eigenspace rotation
        Bcoup = Pperp @ B @ P_old
        Beff = torch.linalg.matrix_norm(Bcoup + Bcoup.T, ord=2).item()
        # morphogenesis step: apply cochain; projection ON removes the coupling block
        if do_proj:
            Bapp = P_old @ B @ P_old + Pperp @ B @ Pperp   # block-diagonal only
        else:
            Bapp = B
        L = sym(L + Bapp)
        P_new, Vk_new, w2, gap2 = spectral_readout(L, k)
        gamma_eff = max(min(gap, gap2), 1e-9)
        # measured end-to-end forgetting on OLD data
        pred_new = head((P_new @ Xeval.T).T)
        forget = (pred_new - pred_ref).norm(dim=1).max().item()
        # theorem bound (worst-case over unit x): Lip_h * 1 * ||P_perp B P_old|| / gamma_eff
        bound = Lip_h * 1.0 * Beff / gamma_eff
        ratio = forget / max(bound, 1e-12)
        max_ratio = max(max_ratio, ratio); worst_forget = max(worst_forget, forget)
        # advance the readout reference frame for the next streamed task
        P_old, Vk_old, gap = P_new, Vk_new, gap2
    return worst_forget, max_ratio

print(f'device={dev}  d={d} k={k} n_tasks={n_tasks} n_eval={n_eval}  Lip_h={Lip_h:.4f}')
print('-'*90)
print(f'{"beta":>6} {"forget_OFF":>12} {"forget_ON":>12} {"ratio_OFF":>12} {"ratio_ON":>12} {"bound_holds":>12}')
betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
all_ok = True; mono = []
for b in betas:
    f_off, r_off = run_stream(b, do_proj=False)
    f_on,  r_on  = run_stream(b, do_proj=True)
    holds = (r_off <= 1.0 + 1e-6) and (r_on <= 1.0 + 1e-6)
    if not holds: all_ok = False
    mono.append(f_off)
    print(f'{b:6.2f} {f_off:12.6f} {f_on:12.6f} {r_off:12.4f} {r_on:12.4f} {str(holds):>12}')
print('-'*90)
print(f'THEOREM bound (forget <= Lip*||x||*||Pperp B Pold||/gamma) holds all beta, both modes: {"PASS" if all_ok else "FAIL"}')
print(f'Forgetting_OFF monotone non-decreasing in beta: {all(mono[i]<=mono[i+1]+1e-9 for i in range(len(mono)-1))}')
f_on_max = max(run_stream(b, do_proj=True)[0] for b in betas)
print(f'Projection ON keeps forgetting ~0 across all beta (max={f_on_max:.2e}): {f_on_max < 1e-6}')
print('Done.')
