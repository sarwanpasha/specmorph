import numpy as np
# Independent re-verification of ALL theory claims in main.tex (PI audit, fresh code).
# Uses only numpy. Fixed seed for reproducibility; many random trials + adversarial cases.
rng = np.random.default_rng(20260716)

def sinTheta(Ua, Ub):
    # principal-angle sin between two orthonormal-column subspaces; ||sinTheta||_2
    # = ||(I-Ua Ua^T) Ub||_2  (largest sin of principal angle)
    Pa = Ua @ Ua.T
    M = (np.eye(Pa.shape[0]) - Pa) @ Ub
    s = np.linalg.svd(M, compute_uv=False)
    return s.max()

def readout_subspace(L, k):
    # invariant subspace of the k SMALLEST eigenvalues (the "read-out")
    w, V = np.linalg.eigh(L)
    idx = np.argsort(w)
    return V[:, idx[:k]], w[idx]

def make_L_with_gap(N, k, gap, rng):
    # random symmetric PSD with an engineered gap after the k-th eigenvalue
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    low = np.sort(rng.uniform(0.0, 0.5, size=k))
    high = np.sort(rng.uniform(0.5 + gap, 1.5 + gap, size=N - k))
    ev = np.concatenate([low, high])
    L = Q @ np.diag(ev) @ Q.T
    L = 0.5 * (L + L.T)
    return L, ev

results = {}

# ---------- THM 1: coupling-block Davis-Kahan, YWS unperturbed-gap form ----------
# claim: ||sinTheta|| <= 2 ||Pperp B Pread|| / gamma   when ||B|| <= gamma/2
fails1 = 0; n1 = 0; ratios = []
for t in range(4000):
    N = int(rng.integers(8, 40)); k = int(rng.integers(1, max(2, N//3)))
    gap = float(rng.uniform(0.3, 3.0))
    L, ev = make_L_with_gap(N, k, gap, rng)
    Uread, w = readout_subspace(L, k)
    gamma = w[k] - w[k-1]   # actual unperturbed gap
    # random symmetric B scaled so ||B|| <= gamma/2
    Braw = rng.standard_normal((N, N)); Braw = 0.5*(Braw+Braw.T)
    nb = np.linalg.norm(Braw, 2)
    scale = (gamma/2) * rng.uniform(0.05, 1.0) / (nb + 1e-12)
    B = scale * Braw
    assert np.linalg.norm(B,2) <= gamma/2 + 1e-9
    Lnew = L + B
    Unew, _ = readout_subspace(Lnew, k)
    st = sinTheta(Uread, Unew)
    Pread = Uread @ Uread.T; Pperp = np.eye(N) - Pread
    coup = np.linalg.norm(Pperp @ B @ Pread, 2)
    bound = 2*coup/gamma
    n1 += 1
    if st > bound + 1e-8: fails1 += 1
    if coup > 1e-9: ratios.append(st/bound)
results['THM1_YWS'] = (fails1, n1, float(np.mean(ratios)), float(np.max(ratios)))

# ---------- COR proj: zero coupling block => sinTheta ~ 0 to first order ----------
# Build B with EXACTLY zero coupling block (block-diagonal wrt Pread/Pperp) and check small rotation
fails_proj = 0; n_proj = 0; maxrot = 0.0
for t in range(2000):
    N = int(rng.integers(8, 30)); k = int(rng.integers(1, max(2, N//3)))
    gap = float(rng.uniform(0.5, 3.0))
    L, ev = make_L_with_gap(N, k, gap, rng)
    Uread, w = readout_subspace(L, k); gamma = w[k]-w[k-1]
    Pread = Uread@Uread.T; Pperp = np.eye(N)-Pread
    Braw = rng.standard_normal((N,N)); Braw=0.5*(Braw+Braw.T)
    # project to block-diagonal: keep only within-block parts
    B = Pread@Braw@Pread + Pperp@Braw@Pperp
    nb=np.linalg.norm(B,2)
    B = B*((gamma/2)*rng.uniform(0.05,0.9)/(nb+1e-12))
    Lnew=L+B; Unew,_=readout_subspace(Lnew,k)
    st=sinTheta(Uread,Unew)
    n_proj+=1; maxrot=max(maxrot,st)
    # first-order zero; allow small 2nd-order term
    if st > 1e-2: fails_proj += 1
results['COR_PROJ'] = (fails_proj, n_proj, maxrot)
print("checkpoint A done", flush=True)
import json
open('RESULT_theory_indep_partA.txt','w').write(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
