#!/usr/bin/env python3
"""Spectral Morphogenesis Bound Test. Exact spec. Pure PyTorch, CPU-only, deterministic, single file."""
import torch, torch.nn as nn, statistics
torch.manual_seed(42)
n, d, k = 12, 2, 2
N = n * d

def build_L():
    L = torch.zeros(N, N)
    I = torch.eye(d)
    for i in range(n):
        u, v = i, (i + 1) % n
        L[u*d:(u+1)*d, u*d:(u+1)*d] += I
        L[v*d:(v+1)*d, v*d:(v+1)*d] += I
        L[u*d:(u+1)*d, v*d:(v+1)*d] -= I
        L[v*d:(v+1)*d, u*d:(u+1)*d] -= I
    return 0.5 * (L + L.T)

def eigs_P_g(Lmat, kk=k):
    ev, evecs = torch.linalg.eigh(Lmat)
    V = evecs[:, :kk]
    P = V @ V.T
    g = (ev[kk] - ev[kk-1]).clamp(min=1e-8).item()
    return P, g, ev

L_old = build_L().detach()
P_old, gamma, ev = eigs_P_g(L_old)
print(f"Base gamma (k={k}) = {gamma:.4f}  lowest evals = {[round(e.item(),4) for e in ev[:k+2]]}")

head = nn.Linear(N, 1, bias=False)
with torch.no_grad():
    head.weight.data /= max(1., torch.linalg.matrix_norm(head.weight.data, 2).item())

def enforce():
    with torch.no_grad():
        sn = torch.linalg.matrix_norm(head.weight, 2).item()
        if sn > 1.: head.weight.data /= sn

X = torch.randn(48, N)
X_eval = X[:24].detach()
x_avg = X_eval.norm(dim=1).mean().item()
opt = torch.optim.Adam(head.parameters(), lr=0.05)
for _ in range(50):
    pred = head((P_old @ X.T).T)
    y = (P_old @ X.T).T.sum(1, keepdim=True) * 0.4
    loss = ((pred - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); enforce()

with torch.no_grad():
    pred_old = head((P_old @ X_eval.T).T)
print(f"Head consolidated. gamma={gamma:.4f}  x_avg={x_avg:.3f}")
print("-" * 100)
print(f"{'beta':>6} {'proj':>4} {'||B||':>8} {'gamma':>7} {'||Pd||':>8} {'Dact':>9} {'Dpred':>9} {'ratio':>7} {'flag':>5}")
print("-" * 100)

betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
all_ok = True
onB = []; onD = []; offB = []; offD = []

for beta in betas:
    for do_proj in (False, True):
        Nnew = N + d
        Lnew = torch.zeros(Nnew, Nnew)
        Lnew[:N, :N] = L_old.clone()
        R0 = (0.5 * torch.randn(d, d) + 0.6 * torch.eye(d)).detach()
        R1 = (0.5 * torch.randn(d, d) + 0.6 * torch.eye(d)).detach()
        s = float(beta)
        for (R, v0) in [(R0, 0), (R1, 1)]:
            blk_n = slice(n*d, (n+1)*d)
            blk_v = slice(v0*d, (v0+1)*d)
            RR = (s * R).T @ (s * R)
            Lnew[blk_n, blk_n] += RR
            Lnew[blk_v, blk_v] += 0.5 * RR
            Lnew[blk_n, blk_v] -= 0.75 * RR
            Lnew[blk_v, blk_n] -= 0.75 * RR
        Lnew = 0.5 * (Lnew + Lnew.T)

        B = Lnew[:N, :N] - L_old
        if do_proj:
            IP = torch.eye(N) - P_old
            B = IP @ B @ IP
            Lnew[:N, :N] = L_old + B
            Lnew = 0.5 * (Lnew + Lnew.T)
        Bnorm = torch.linalg.matrix_norm(B, 2).item()

        Pnew, _, _ = eigs_P_g(Lnew[:N, :N])
        Pdiff = torch.linalg.matrix_norm(P_old - Pnew, 2).item()

        with torch.no_grad():
            pred_new = head((Pnew @ X_eval.T).T)
        Dact = (pred_new - pred_old).abs().mean().item()
        Dpred = 1.0 * x_avg * (Bnorm / max(gamma, 1e-8))
        ratio = Dact / max(Dpred, 1e-12)
        flag = "PASS" if Dact <= Dpred + 1e-4 else "FAIL"
        if flag == "FAIL": all_ok = False

        print(f"{beta:6.2f} {'ON' if do_proj else 'OFF':>4} {Bnorm:8.4f} {gamma:7.4f} {Pdiff:8.4f} {Dact:9.5f} {Dpred:9.5f} {ratio:7.3f} {flag:>5}")
        if do_proj:
            onB.append(Bnorm); onD.append(Dact)
        else:
            offB.append(Bnorm); offD.append(Dact)

print("-" * 100)
print(f"Overall bound holds every row: {'PASS' if all_ok else 'FAIL'}")
print(f"Proj ON reduced mean ||B|| (beta>0): {statistics.mean(offB[1:]) > statistics.mean(onB[1:])}")
print(f"Proj ON reduced mean Dact (beta>0): {statistics.mean(offD[1:]) > statistics.mean(onD[1:])}")
print("Done.")
