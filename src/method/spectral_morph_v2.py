#!/usr/bin/env python3
import torch, torch.nn as nn, statistics
torch.manual_seed(42)
n, d, k = 12, 2, 2
N = n * d

def build_L():
    L = torch.zeros(N, N); I = torch.eye(d)
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
    return V @ V.T, (ev[kk] - ev[kk-1]).clamp(min=1e-8).item(), ev

def opn(M):
    return torch.linalg.matrix_norm(M, 2).item()

L_old = build_L().detach()
P_old, gamma, ev = eigs_P_g(L_old)
Pp = torch.eye(N) - P_old
print("Base gamma (k=%d) = %.4f  low evals = %s" % (k, gamma, [round(e.item(),4) for e in ev[:k+2]]))
head = nn.Linear(N, 1, bias=False)
with torch.no_grad():
    head.weight.data /= max(1., opn(head.weight.data))

def enforce():
    with torch.no_grad():
        sn = opn(head.weight)
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
print("Head consolidated. gamma=%.4f  x_avg=%.3f" % (gamma, x_avg))
print("-"*120)
print("%5s %4s %8s %9s %8s %8s %9s %9s %9s %7s %7s %5s" % ("beta","proj","||B||","||Beff||","gam_eff","||Pd||","Dact","Dpr_w","Dpr_d","rw","rd","flag"))
print("-"*120)
betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
all_ok = True
onBe=[]; onD=[]; offBe=[]; offD=[]
for beta in betas:
    for do_proj in (False, True):
        Lnew = torch.zeros(N+d, N+d); Lnew[:N,:N] = L_old.clone()
        R0 = (0.5*torch.randn(d,d)+0.6*torch.eye(d)).detach()
        R1 = (0.5*torch.randn(d,d)+0.6*torch.eye(d)).detach()
        s = float(beta)
        for (R, v0) in [(R0,0),(R1,1)]:
            bn = slice(n*d,(n+1)*d); bv = slice(v0*d,(v0+1)*d)
            RR = (s*R).T @ (s*R)
            Lnew[bn,bn] += RR; Lnew[bv,bv] += 0.5*RR
            Lnew[bn,bv] -= 0.75*RR; Lnew[bv,bn] -= 0.75*RR
        Lnew = 0.5*(Lnew+Lnew.T)
        B = Lnew[:N,:N] - L_old
        if do_proj:
            B = Pp @ B @ Pp
            Lnew[:N,:N] = L_old + B; Lnew = 0.5*(Lnew+Lnew.T)
        Bnorm = opn(B)
        Beff = opn(Pp @ B @ P_old)
        Ls = L_old + P_old @ B @ P_old
        _, gamma_eff, _ = eigs_P_g(0.5*(Ls+Ls.T))
        Pnew, _, _ = eigs_P_g(Lnew[:N,:N])
        Pd = opn(P_old - Pnew)
        with torch.no_grad():
            pred_new = head((Pnew @ X_eval.T).T)
        Dact = (pred_new - pred_old).abs().mean().item()
        Dpr_d = ((Pnew - P_old) @ X_eval.T).T.norm(dim=1).mean().item()
        Dpr_w = 1.0 * x_avg * (Beff / max(gamma_eff,1e-8))
        rw = Dact / max(Dpr_w,1e-12); rd = Dact / max(Dpr_d,1e-12)
        flag = "PASS" if Dact <= Dpr_w + 1e-4 else "FAIL"
        if flag=="FAIL": all_ok=False
        print("%5.2f %4s %8.4f %9.4f %8.4f %8.4f %9.5f %9.5f %9.5f %7.3f %7.3f %5s" % (beta, ("ON" if do_proj else "OFF"), Bnorm, Beff, gamma_eff, Pd, Dact, Dpr_w, Dpr_d, rw, rd, flag))
        if do_proj: onBe.append(Beff); onD.append(Dact)
        else: offBe.append(Beff); offD.append(Dact)
print("-"*120)
print("Overall THEOREM bound (Dact <= Dpr_w) holds every row: %s" % ("PASS" if all_ok else "FAIL"))
print("Proj ON drives mean ||Beff|| (beta>0) ~0: %s" % (statistics.mean(onBe[1:]) < 1e-4 and statistics.mean(offBe[1:]) > statistics.mean(onBe[1:])))
print("Proj ON drives mean Dact   (beta>0) ~0: %s" % (statistics.mean(onD[1:]) < 1e-4 and statistics.mean(offD[1:]) > statistics.mean(onD[1:])))
print("Done.")
