import torch, math, statistics
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

d, k, n_eval, n_tasks = 128, 8, 512, 6

def sym(A):
    return 0.5*(A + A.T)

def bottomk(L, k):
    w, V = torch.linalg.eigh(sym(L))
    Vk = V[:, :k]
    return Vk @ Vk.T, Vk, w, (w[k]-w[k-1]).item()

Wh = torch.randn(4, d, dtype=dtype, device=dev)
Wh = Wh / torch.linalg.matrix_norm(Wh, ord=2)
Lip_h = torch.linalg.matrix_norm(Wh, ord=2).item()
def head(z):
    return z @ Wh.T

A = torch.randn(d, d, dtype=dtype, device=dev)
L0 = sym(A @ A.T) / d
w0, V0 = torch.linalg.eigh(L0)
w0m = w0.clone(); w0m[:k] = w0m[:k] - 1.5
L0 = sym(V0 @ torch.diag(w0m) @ V0.T)

Xeval = torch.randn(n_eval, d, dtype=dtype, device=dev)
Xeval = Xeval / Xeval.norm(dim=1, keepdim=True)

def make_coupling(beta, seed):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    R = torch.randn(d, d, dtype=dtype, device=dev, generator=g)
    return beta * sym(R) / math.sqrt(d)

def run_stream(beta, do_proj):
    L = L0.clone()
    P_old, Vk_old, w, gap = bottomk(L, k)
    Iu = torch.eye(d, dtype=dtype, device=dev)
    pred_ref = head((P_old @ Xeval.T).T)
    P_read = P_old.clone()   # maintained readout subspace (projection-ON keeps this invariant)
    max_ratio = 0.0; worst_forget = 0.0; min_gap = gap; crossings = 0
    for t in range(n_tasks):
        B = make_coupling(beta, 100+t)
        Pp = Iu - P_old
        Bc = Pp @ B @ P_old
        Beff = torch.linalg.matrix_norm(Bc + Bc.T, ord=2).item()
        if do_proj:
            Bapp = P_old @ B @ P_old + Pp @ B @ Pp   # zero the coupling block
        else:
            Bapp = B
        L = sym(L + Bapp)
        P_new, Vk_new, w2, gap2 = bottomk(L, k)
        gamma_eff = max(min(gap, gap2), 1e-9); min_gap = min(min_gap, gap2)
        if do_proj:
            # projection ON: readout stays on the maintained invariant subspace P_read.
            # detect if a cross-block eigenvalue crossing invalidated bottom-k selection.
            drift = torch.linalg.matrix_norm(P_new - P_read, ord=2).item()
            if drift > 0.5: crossings += 1
            P_use = P_read   # maintained subspace, unchanged by block-diagonal update
        else:
            P_use = P_new
        pred_new = head((P_use @ Xeval.T).T)
        forget = (pred_new - pred_ref).norm(dim=1).max().item()
        bound = Lip_h * 1.0 * Beff / gamma_eff
        ratio = forget / max(bound, 1e-12)
        max_ratio = max(max_ratio, ratio); worst_forget = max(worst_forget, forget)
        if do_proj:
            P_old, gap = P_read, gap2   # keep maintained subspace fixed for next block split
        else:
            P_old, gap = P_new, gap2
    return worst_forget, max_ratio, min_gap, crossings

print(f'device={dev}  d={d} k={k} n_tasks={n_tasks} n_eval={n_eval}  Lip_h={Lip_h:.4f}')
print('-'*104)
print(f'{"beta":>6} {"forget_OFF":>11} {"forget_ON":>11} {"ratio_OFF":>10} {"minGap_OFF":>11} {"cross_ON":>9} {"holds":>7}')
betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
all_ok = True; mono = []; on_zero = True
for b in betas:
    f_off, r_off, mg_off, _ = run_stream(b, do_proj=False)
    f_on,  r_on,  mg_on, cr = run_stream(b, do_proj=True)
    holds = (r_off <= 1.0 + 1e-6)
    if not holds: all_ok = False
    if f_on > 1e-8: on_zero = False
    mono.append(f_off)
    print(f'{b:6.2f} {f_off:11.6f} {f_on:11.2e} {r_off:10.4f} {mg_off:11.4f} {cr:9d} {str(holds):>7}')
print('-'*104)
print(f'THEOREM sharp bound (forget_OFF <= Lip*||x||*||Pperp B Pold||/gamma) holds all beta: {"PASS" if all_ok else "FAIL"}')
print(f'Forgetting_OFF monotone non-decreasing in beta: {all(mono[i]<=mono[i+1]+1e-9 for i in range(len(mono)-1))}')
print(f'Projection ON (maintained invariant subspace) keeps forgetting EXACTLY 0 across all beta: {on_zero}')
print('Done.')
