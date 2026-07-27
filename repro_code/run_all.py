"""
Streamlined reproduction of all 4 claims.
Runs faster experiments with modest parameters.
"""
import os, sys, json, time, numpy as np, torch
from scipy import stats
from scipy.special import gamma as gamma_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

torch.set_default_dtype(torch.float64)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

p, M_star, sigma = 5, 5, 1.0
alpha_theory = (p + 1) / 2.0  # = 3.0

def gen_true(seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(M_star, p)
    for i in range(1, M_star):
        for j in range(i):
            W1[i] -= np.dot(W1[i], W1[j]) / (np.dot(W1[j], W1[j]) + 1e-10) * W1[j]
    W1 = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    w2 = np.array([(-1)**i * (1.0 + 0.5*i) for i in range(M_star)])
    for m in range(M_star): W1[m] *= np.abs(w2[m])
    return W1, w2

def gen_data(W1, w2, n, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.maximum(X @ W1.T, 0) @ w2 + sigma * rng.randn(n)
    return X, y

W1_star, w2_star = gen_true(42)

def hmc_sample(X, y, M, n_samp=300, n_warm=150, step=5e-4, n_chains=2, seed=0):
    n, p = X.shape
    d = M * (p + 1)
    X_t, y_t = torch.tensor(X), torch.tensor(y)

    def logp(w):
        W1 = w[:M*p].reshape(M, p); w2 = w[M*p:]
        f = torch.relu(X_t @ W1.T) @ w2
        return -0.5*torch.sum((y_t-f)**2) - torch.sum(w**2)

    def grad_fn(w):
        w.requires_grad_(True)
        return torch.autograd.grad(logp(w), w)[0]

    all_samps = []
    for ch in range(n_chains):
        torch.manual_seed(seed + ch*1000)
        w = torch.randn(d) * 0.2
        for i in range(n_warm):
            w = w.clone().detach().requires_grad_(True)
            p_mom = torch.randn_like(w)
            w_new, p_new = w.clone(), p_mom.clone()
            g = grad_fn(w_new)
            p_new = p_new + 0.5*step*g
            for _ in range(8):
                w_new = (w_new + step*p_new).detach().requires_grad_(True)
                p_new = p_new + step*grad_fn(w_new)
            w_new = (w_new + step*p_new).detach().requires_grad_(True)
            p_new = p_new + 0.5*step*grad_fn(w_new)
            lp_old, lp_new = logp(w), logp(w_new)
            if np.log(np.random.rand()) < (lp_new - lp_old).item():
                w = w_new.detach()
            if i % 50 == 0 and i > 0:
                ar = np.clip(np.exp((lp_new-lp_old).item()), 0, 1)
                step *= 1.2 if ar > 0.8 else (0.8 if ar < 0.2 else 1.0)
                step = np.clip(step, 1e-6, 0.1)
        samps = []
        for i in range(n_samp * 3):
            w = w.clone().detach().requires_grad_(True)
            p_mom = torch.randn_like(w)
            w_new, p_new = w.clone(), p_mom.clone()
            g = grad_fn(w_new)
            p_new = p_new + 0.5*step*g
            for _ in range(8):
                w_new = (w_new + step*p_new).detach().requires_grad_(True)
                p_new = p_new + step*grad_fn(w_new)
            w_new = (w_new + step*p_new).detach().requires_grad_(True)
            p_new = p_new + 0.5*step*grad_fn(w_new)
            lp_old, lp_new = logp(w), logp(w_new)
            if np.log(np.random.rand()) < (lp_new-lp_old).item():
                w = w_new.detach()
            if i % 3 == 0: samps.append(w.clone().numpy())
        all_samps.append(np.array(samps))
        print(f'  Chain {ch+1}: {len(samps)} samples')
    return np.concatenate(all_samps, axis=0)

# ========== CLAIM 1 ==========
print("="*60); print("CLAIM 1"); print("="*60)
M = 10; n = 500
X, y = gen_data(W1_star, w2_star, n, 42)
samps = hmc_sample(X, y, M, n_samp=300, n_warm=150, step=5e-4, n_chains=2, seed=42)
aw = samps

all_c_1 = []
for w in aw:
    W1 = w[:M*p].reshape(M, p)
    for m in range(M):
        sims = [np.dot(W1[m], W1_star[ms])/(np.linalg.norm(W1[m])*np.linalg.norm(W1_star[ms])+1e-10) for ms in range(M_star)]
        ms = np.argmax(sims)
        c = np.linalg.norm(W1[m])**2 / (np.linalg.norm(W1_star[ms])**2 + 1e-10)
        all_c_1.append(c)
all_c_1 = np.array(all_c_1)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
k = M // M_star
ax.hist(all_c_1, bins=50, density=True, alpha=0.6, label='Empirical', color='steelblue')
x = np.linspace(0.001, 1.0, 200)
ax.plot(x, stats.beta.pdf(x, alpha_theory, (k-1)*alpha_theory), 'r-', lw=2,
         label=f'Beta({alpha_theory:.1f},{(k-1)*alpha_theory:.1f})')
ax.set(xlabel='c_m', ylabel='Density', title=f'Claim 1: Splitting Coeffs (M={M}, n={n})')
ax.legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR/'claim1.png', dpi=150); plt.close()
print(f'  Mean c_m = {np.mean(all_c_1):.4f} (theory: {1/k:.4f})')

# ========== CLAIM 2 ==========
print("="*60); print("CLAIM 2"); print("="*60)
n_vals = [64, 128, 256, 512]
M_vals = [5, 10]
results2 = {}
for M in M_vals:
    tr_vals, rmse_vals = [], []
    for n in n_vals:
        X, y = gen_data(W1_star, w2_star, n, 42)
        aw = hmc_sample(X, y, M, n_samp=200, n_warm=100, step=5e-4, n_chains=2, seed=42)
        tr = np.sum(np.var(aw, axis=0)) / (M*(p+1))
        # RMSE on test
        Xt, yt = gen_data(W1_star, w2_star, 200, 999)
        Xtt = torch.tensor(Xt)
        preds = [(torch.relu(Xtt@torch.tensor(w[:M*p].reshape(M,p)))@torch.tensor(w[M*p:])).numpy() for w in aw[:100]]
        rmse = np.sqrt(np.mean((np.mean(preds, axis=0)-yt)**2))
        tr_vals.append(tr); rmse_vals.append(rmse)
        print(f'  M={M}, n={n}: tr/d={tr:.6f}, RMSE={rmse:.4f}')
    results2[f'M={M}'] = {'n': n_vals, 'tr': tr_vals, 'rmse': rmse_vals}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for M in M_vals:
    axes[0].plot(results2[f'M={M}']['n'], results2[f'M={M}']['rmse'], 'o-', label=f'M={M}')
    axes[1].plot(results2[f'M={M}']['n'], results2[f'M={M}']['tr'], 's-', label=f'M={M}')
axes[0].set(xlabel='n', ylabel='RMSE', title='Claim 2: Prediction', xscale='log'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set(xlabel='n', ylabel='tr(Cov)/d', title='Claim 2: Parameter Uncertainty', xscale='log', yscale='log'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(RESULTS_DIR/'claim2.png', dpi=150); plt.close()

# ========== CLAIM 3 ==========
print("="*60); print("CLAIM 3"); print("="*60)
n = 1000
M_vals3 = [10, 20]
results3 = {}
fig, axes = plt.subplots(2, len(M_vals3), figsize=(5*len(M_vals3), 9))
for mi, M in enumerate(M_vals3):
    k = M // M_star
    X, y = gen_data(W1_star, w2_star, n, 42)
    aw = hmc_sample(X, y, M, n_samp=300, n_warm=150, step=3e-4, n_chains=2, seed=42)
    all_c3, all_s3 = [], []
    for w in aw:
        W1 = w[:M*p].reshape(M, p); w2 = w[M*p:]
        for m in range(M):
            sims = [np.dot(W1[m], W1_star[ms])/(np.linalg.norm(W1[m])*np.linalg.norm(W1_star[ms])+1e-10) for ms in range(M_star)]
            ms = np.argmax(sims)
            c = np.linalg.norm(W1[m])**2/(np.linalg.norm(W1_star[ms])**2+1e-10)
            om = np.concatenate([W1[m],[w2[m]]]); os = np.concatenate([W1_star[ms],[w2_star[ms]]])
            s = np.dot(om,os)/(np.dot(os,os)+1e-10)
            all_c3.append(c); all_s3.append(s)
    all_c3, all_s3 = np.array(all_c3), np.array(all_s3)
    def mu_k_a(k,a): return gamma_fn(a+0.5)*gamma_fn(k*a)/(gamma_fn(a)*gamma_fn(k*a+0.5))
    t = [1/k, mu_k_a(k,alpha_theory), 1/k]
    e = [np.mean(all_c3), np.mean(all_s3), np.mean(all_s3**2)]
    results3[f'M={M}'] = {'k':k, 'theory':t, 'empirical':e}
    print(f'  M={M}, k={k}: Ec={e[0]:.4f}/{t[0]:.4f}, Es={e[1]:.4f}/{t[1]:.4f}, Es2={e[2]:.4f}/{t[2]:.4f}')
    axes[0,mi].hist(all_c3, bins=50, density=True, alpha=0.6, color='steelblue', label='Empirical')
    axes[0,mi].plot(np.linspace(0.001,1,200), stats.beta.pdf(np.linspace(0.001,1,200), alpha_theory, (k-1)*alpha_theory), 'r-', lw=2)
    axes[0,mi].set(title=f'M={M}, k={k}', xlabel='c_m', ylabel='Density'); axes[0,mi].legend(fontsize=8)
    cats = ['E[c]','E[s]','E[s²]']
    axes[1,mi].bar(np.arange(3)-0.15, t, 0.3, label='Theory', color='coral')
    axes[1,mi].bar(np.arange(3)+0.15, e, 0.3, label='Empirical', color='steelblue')
    axes[1,mi].set_xticks(np.arange(3)); axes[1,mi].set_xticklabels(cats)
    axes[1,mi].set(title=f'Theorem 1 (M={M})'); axes[1,mi].legend(); axes[1,mi].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(RESULTS_DIR/'claim3.png', dpi=150); plt.close()
max_err = max(abs(results3[f'M={M}']['empirical'][i]-results3[f'M={M}']['theory'][i]) for M in M_vals3 for i in range(3))
print(f'  Max error: {max_err:.4f}')

# ========== CLAIM 4 ==========
print("="*60); print("CLAIM 4"); print("="*60)
M = 10; n_vals = [128, 256, 512, 1024]
tr4, vc4, rmse4 = [], [], []
for n in n_vals:
    X, y = gen_data(W1_star, w2_star, n, 42)
    aw = hmc_sample(X, y, M, n_samp=250, n_warm=120, step=3e-4, n_chains=2, seed=42)
    tr = np.sum(np.var(aw, axis=0)) / (M*(p+1))
    all_c4 = []
    for w in aw:
        W1 = w[:M*p].reshape(M,p)
        for m in range(M):
            sims = [np.dot(W1[m],W1_star[ms])/(np.linalg.norm(W1[m])*np.linalg.norm(W1_star[ms])+1e-10) for ms in range(M_star)]
            ms = np.argmax(sims)
            all_c4.append(np.linalg.norm(W1[m])**2/(np.linalg.norm(W1_star[ms])**2+1e-10))
    Xt, yt = gen_data(W1_star, w2_star, 200, 999)
    Xtt = torch.tensor(Xt)
    preds = [(torch.relu(Xtt@torch.tensor(w[:M*p].reshape(M,p)))@torch.tensor(w[M*p:])).numpy() for w in aw[:100]]
    rmse = np.sqrt(np.mean((np.mean(preds,axis=0)-yt)**2))
    tr4.append(tr); vc4.append(np.var(all_c4)); rmse4.append(rmse)
    print(f'  n={n}: tr/d={tr:.6f}, var(c)={np.var(all_c4):.6f}, RMSE={rmse:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(n_vals, tr4, 'o-', color='steelblue')
axes[0].set(xlabel='n', ylabel='tr(Cov)/d', title='Claim 4: Parameter Uncertainty', xscale='log', yscale='log'); axes[0].grid(alpha=0.3)
axes[1].plot(n_vals, vc4, 's-', color='coral')
axes[1].set(xlabel='n', ylabel='Var(c_m)', title='Claim 4: Splitting Variance', xscale='log'); axes[1].grid(alpha=0.3)
axes[2].plot(n_vals, rmse4, 'o-', color='steelblue')
axes[2].axhline(y=sigma, color='gray', ls='--', label=f'σ={sigma}')
axes[2].set(xlabel='n', ylabel='RMSE', title='Claim 4: Function Convergence', xscale='log'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(RESULTS_DIR/'claim4.png', dpi=150); plt.close()

# Save results
results = {
    'claim1': {'mean_c': float(np.mean(all_c_1)), 'theory_Ec': 1/k},
    'claim2': {str(k): v for k,v in results2.items()},
    'claim3': {str(k): {'k':v['k'], 'theory':[float(x) for x in v['theory']], 'empirical':[float(x) for x in v['empirical']]} for k,v in results3.items()},
    'claim4': {'n': n_vals, 'tr_d': [float(x) for x in tr4], 'var_c': [float(x) for x in vc4], 'rmse': [float(x) for x in rmse4]}
}
json.dump(results, open(RESULTS_DIR/'all_results.json','w'), indent=2)
print("\nDone! Results saved.")
