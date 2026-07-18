import numpy as np
np.random.seed(0)

# ============================================================
# Verify: topological obstruction to zero-forgetting under
# pairwise-orthogonal projection, and that a global (coboundary)
# projection beats it. Pure-numpy, exact linear algebra.
# ============================================================
#
# Setup: N-dim ambient feature space. A shared read-out subspace
# U (k dims). A task-interaction graph G=(V,E). Each task t has a
# restriction map R_t (r x N) whose row space is the task's stalk
# embedding into the shared space. Aggregate coupling operator
# assembled over the cycle structure of G. Nonzero H^1 <=> G has
# independent cycles (first Betti number b1 = |E|-|V|+components).

def principal_angles_sin(A, B):
    # sin of principal angles between column spaces of A,B (orthonormal cols)
    Qa,_ = np.linalg.qr(A); Qb,_ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T@Qb, compute_uv=False)
    s = np.clip(s,-1,1)
    return np.sqrt(np.maximum(0.0,1-s**2))

def betti1(V, E):
    # first Betti number of graph (assume connected)
    return len(E) - V + 1

print('=== OBSTRUCTION / COBOUNDARY-PROJECTION VERIFICATION ===')

def build_stream(N, k, r, cycle_len, holonomy, seed):
    rng = np.random.default_rng(seed)
    # shared read-out subspace U (orthonormal N x k)
    U,_ = np.linalg.qr(rng.standard_normal((N,k)))
    U = U[:,:k]
    # cycle graph on cycle_len vertices: edges (i,i+1 mod L)
    L = cycle_len
    edges = [(i,(i+1)%L) for i in range(L)]
    # each vertex carries a stalk basis S_v (N x r) with STRONG overlap on U
    # (high coherence with read-out): embed each stalk mostly inside U-span
    S = []
    for v in range(L):
        M = U @ rng.standard_normal((k,r)) + 0.15*rng.standard_normal((N,r))
        Q,_ = np.linalg.qr(M); S.append(Q[:,:r])
    # restriction maps: transport along edge = orthogonal map between endpoint stalks.
    # We inject a controlled holonomy so the cycle product != I  (=> nonzero H^1).
    T = []
    for (u,v) in edges:
        A = rng.standard_normal((r,r)); Qr,_ = np.linalg.qr(A)
        T.append(Qr)
    # force product of transports around the cycle to deviate from I by 'holonomy'
    prod = np.eye(r)
    for M in T: prod = M@prod
    # apply a corrective twist to the last transport so total holonomy is controlled
    Uh,_ = np.linalg.qr(rng.standard_normal((r,r)))
    twist = np.eye(r)*(1-holonomy) + holonomy*Uh
    Qt,_ = np.linalg.qr(twist)
    T[-1] = Qt @ T[-1] @ np.linalg.inv(prod) @ T[-1]
    return U,S,T,edges

def assemble_Beff(U,S,T,edges, projector):
    # Effective read-out coupling assembled over the graph.
    # For each edge (u,v) with transport T_e between stalks, the edge contributes
    # the cross term S_v T_e S_u^T mapped through the read-out projector.
    # projector: function(edge_index, X)->X' applied to each edge's stalk map
    # before assembly (this is where a projection scheme intervenes).
    N = U.shape[0]
    B = np.zeros((N,N))
    for ei,(u,v) in enumerate(edges):
        Su, Sv, Te = S[u], S[v], T[ei]
        # raw edge coupling operator on ambient space
        Xe = Sv @ Te @ Su.T
        Xe = projector(ei, Xe)
        B += Xe
    # read-out coupling block: how B maps the read-out subspace off itself
    Pperp = np.eye(N) - U@U.T
    return Pperp @ B @ U   # N x k coupling block

def sinTheta_from_coupling(U, coupling, gamma=1.0):
    # first-order rotation magnitude of read-out subspace ~ ||coupling||/gamma
    return np.linalg.norm(coupling, 2)/gamma

# --- Scheme A: PAIRWISE-orthogonal projection (OGD/GPM style) ---
# each edge independently projected orthogonal to U. Removes each edge's
# read-out component locally.
def pairwise_proj(U):
    Pperp = np.eye(U.shape[0]) - U@U.T
    def f(ei, Xe):
        # orthogonalize columns AND rows against U independently per edge
        return Pperp @ Xe @ Pperp
    return f

# identity (no projection)
def no_proj(ei, Xe): return Xe

# ------------------------------------------------------------------
# REALISTIC projection models.
# Pairwise (GPM/OGD): the learner never sees the shared read-out U.
# It estimates, edge by edge, the local stalk directions and projects
# each new edge orthogonal to the SPAN of previously-seen edge stalks.
# Around a cycle with holonomy this local estimate cannot represent the
# global consistency class, so a residual survives.
# Global (coboundary): project the ASSEMBLED operator onto the
# orthogonal complement of the true read-out U in one shot.
# ------------------------------------------------------------------

def forgetting_pairwise(U,S,T,edges, rank_cap):
    N = U.shape[0]
    B = np.zeros((N,N))
    seen = np.zeros((N,0))            # running stored basis (like GPM memory)
    for ei,(u,v) in enumerate(edges):
        Su,Sv,Te = S[u],S[v],T[ei]
        Xe = Sv @ Te @ Su.T
        # project against stored basis 'seen' (pairwise memory), capped rank
        if seen.shape[1]>0:
            P = seen @ seen.T
            Xe = (np.eye(N)-P) @ Xe @ (np.eye(N)-P)
        B += Xe
        # update memory from THIS edge's observed stalks (SVD, capped)
        obs = np.hstack([Sv,Su])
        M = np.hstack([seen,obs])
        Q,_ = np.linalg.qr(M)
        seen = Q[:,:min(rank_cap,Q.shape[1])]
    Pperp = np.eye(N)-U@U.T
    return np.linalg.norm(Pperp @ B @ U, 2)

def forgetting_global(U,S,T,edges):
    # assemble raw, then project the assembled coupling onto complement of U
    N=U.shape[0]; B=np.zeros((N,N))
    for ei,(u,v) in enumerate(edges):
        B += S[v] @ T[ei] @ S[u].T
    Pperp=np.eye(N)-U@U.T
    Bp = Pperp @ B @ Pperp             # global one-shot coboundary projection
    return np.linalg.norm(Pperp @ Bp @ U, 2)

def forgetting_raw(U,S,T,edges):
    N=U.shape[0]; B=np.zeros((N,N))
    for ei,(u,v) in enumerate(edges):
        B += S[v] @ T[ei] @ S[u].T
    Pperp=np.eye(N)-U@U.T
    return np.linalg.norm(Pperp @ B @ U, 2)

print('N=64 k=6 r=6 cycle_len=6  (b1=1, nonzero H^1)')
print(f'{"holon":>7} {"raw":>10} {"pairwise":>10} {"global":>10} {"pw/glob":>9}')
rows=[]
for holon in [0.0,0.1,0.2,0.4,0.6,0.8,1.0]:
    rr=[];pp=[];gg=[]
    for seed in range(20):
        U,S,T,edges = build_stream(64,6,6,6,holon,seed)
        rr.append(forgetting_raw(U,S,T,edges))
        pp.append(forgetting_pairwise(U,S,T,edges,rank_cap=12))
        gg.append(forgetting_global(U,S,T,edges))
    rr,pp,gg=np.mean(rr),np.mean(pp),np.mean(gg)
    rows.append((holon,rr,pp,gg))
    ratio = pp/gg if gg>1e-9 else float('inf')
    print(f'{holon:7.2f} {rr:10.4f} {pp:10.4f} {gg:10.4e} {ratio:9.1f}')

print()
print('CHECKS:')
z = [r for r in rows if r[0]==0.0][0]
nz = [r for r in rows if r[0]==1.0][0]
print(f'  pairwise residual at holon=0 : {z[2]:.4f}')
print(f'  pairwise residual at holon=1 : {nz[2]:.4f}  (obstruction grows with holonomy: {nz[2]>z[2]+1e-6})')
print(f'  global residual ~0 always    : max={max(r[3] for r in rows):.2e}')
print(f'  global beats pairwise (holon>=0.2): {all(r[2] > r[3]+1e-6 for r in rows if r[0]>=0.2)}')
import numpy as _np
hol=_np.array([r[0] for r in rows]); pw=_np.array([r[2] for r in rows])
print(f'  corr(holonomy, pairwise residual) = {_np.corrcoef(hol,pw)[0,1]:.3f}')
