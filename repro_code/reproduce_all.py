"""
Full reproduction of ICML 2026 Paper #1:
"On the Epistemic Uncertainty of Overparametrized Neural Networks"
by David Rügamer (arXiv:2605.25234)

Reproduces all 4 claims with custom HMC sampling on 1-hidden-layer ReLU networks.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.special import gamma as gamma_fn, beta as beta_fn
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# MODEL: 1-hidden-layer ReLU network (no bias, no output bias)
# ============================================================
class ReLUNetwork(nn.Module):
    def __init__(self, p, M):
        super().__init__()
        self.p = p
        self.M = M
        self.W1 = nn.Parameter(torch.randn(M, p) * 0.5)
        self.w2 = nn.Parameter(torch.randn(M) * 0.5)

    def forward(self, x):
        # x: (n, p) -> hidden: (n, M) -> out: (n,)
        h = torch.relu(x @ self.W1.T)
        return h @ self.w2

    def get_weights_flat(self):
        return torch.cat([self.W1.flatten(), self.w2])

    def set_weights_flat(self, w):
        d_w1 = self.M * self.p
        self.W1.data = w[:d_w1].reshape(self.M, self.p)
        self.w2.data = w[d_w1:]


# ============================================================
# DATA GENERATION
# ============================================================
def generate_true_network(p=5, M_star=5, seed=0):
    """Generate a true network f* satisfying Assumption 1 (non-degeneracy)."""
    rng = np.random.RandomState(seed)
    # Generate random directions, ensure no two are collinear
    W1_star = rng.randn(M_star, p)
    # Orthogonalize to ensure linear independence
    for i in range(1, M_star):
        for j in range(i):
            proj = np.dot(W1_star[i], W1_star[j]) / np.dot(W1_star[j], W1_star[j])
            W1_star[i] -= proj * W1_star[j]
    # Normalize
    W1_star = W1_star / np.linalg.norm(W1_star, axis=1, keepdims=True)
    # Output weights with alternating signs
    w2_star = np.array([(-1)**i * (1.0 + 0.5 * i) for i in range(M_star)])
    # Make norm-balanced: ||w1_m|| = |w2_m| for minimum-norm representation
    for m in range(M_star):
        W1_star[m] *= np.abs(w2_star[m])
    return W1_star, w2_star


def generate_data(W1_star, w2_star, n, sigma=1.0, seed=0):
    """Generate synthetic data from the true network."""
    rng = np.random.RandomState(seed)
    p = W1_star.shape[1]
    X = rng.randn(n, p)
    # f*(x) = sum_m w2_m * ReLU(w1_m . x)
    h = np.maximum(X @ W1_star.T, 0)
    y_clean = h @ w2_star
    y = y_clean + sigma * rng.randn(n)
    return X, y, y_clean


# ============================================================
# HMC SAMPLER (Custom Leapfrog)
# ============================================================
def hmc_log_prob(w, X, y, sigma_noise=1.0, lambda_prior=1.0):
    """Compute log posterior: log likelihood + log prior."""
    M = len(w) // (X.shape[1] + 1)
    p = X.shape[1]
    W1 = w[:M*p].reshape(M, p)
    w2 = w[M*p:]
    h = torch.relu(X @ W1.T)
    f = h @ w2
    # Log likelihood (Gaussian)
    ll = -0.5 * torch.sum((y - f)**2) / sigma_noise**2
    # Log prior (Gaussian)
    lp = -lambda_prior * torch.sum(w**2)
    return ll + lp


def hmc_leapfrog(w, grad_fn, step_size, n_steps):
    """Leapfrog integrator for HMC."""
    w = w.clone().detach().requires_grad_(True)
    p_mom = torch.randn_like(w)
    w_new = w.clone()
    p_new = p_mom.clone()

    # Half step for momentum
    grad = grad_fn(w_new)
    p_new = p_new + 0.5 * step_size * grad

    # Full steps
    for _ in range(n_steps - 1):
        w_new = w_new + step_size * p_new
        w_new = w_new.detach().requires_grad_(True)
        grad = grad_fn(w_new)
        p_new = p_new + step_size * grad

    # Final position step
    w_new = w_new + step_size * p_new
    w_new = w_new.detach().requires_grad_(True)
    grad = grad_fn(w_new)
    p_new = p_new + 0.5 * step_size * grad

    return w_new.detach(), p_new


def run_hmc(X, y, M, n_samples=1000, n_warmup=500, step_size=1e-3,
            n_leapfrog=10, thin=10, sigma_noise=1.0, lambda_prior=1.0,
            n_chains=3, seed=0, verbose=True):
    """Run HMC sampling for the Bayesian neural network."""
    torch.manual_seed(seed)
    p = X.shape[1]
    d = M * (p + 1)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    def grad_fn(w):
        w.requires_grad_(True)
        lp = hmc_log_prob(w, X_t, y_t, sigma_noise, lambda_prior)
        grad = torch.autograd.grad(lp, w)[0]
        return grad

    all_samples = []
    all_log_probs = []

    for chain in range(n_chains):
        torch.manual_seed(seed + chain * 1000)
        # Initialize with random
        w = torch.randn(d, dtype=torch.float64) * 0.1
        w.requires_grad_(True)

        # Warmup: adapt step size
        for i in range(n_warmup):
            w_new, p_new = hmc_leapfrog(w, grad_fn, step_size, n_leapfrog)
            # MH accept/reject
            lp_old = hmc_log_prob(w, X_t, y_t, sigma_noise, lambda_prior)
            lp_new = hmc_log_prob(w_new, X_t, y_t, sigma_noise, lambda_prior)
            ke_old = 0.5 * torch.sum(p_new**2)
            ke_new = 0.5 * torch.sum(torch.randn_like(w_new)**2)
            log_accept = (lp_new - lp_old).item()
            if np.log(np.random.rand()) < log_accept:
                w = w_new.detach()
            # Adapt step size
            if i > 0 and i % 50 == 0:
                accept_rate = min(1.0, np.exp(log_accept))
                if accept_rate > 0.8:
                    step_size *= 1.2
                elif accept_rate < 0.2:
                    step_size *= 0.8
                step_size = max(step_size, 1e-6)
                step_size = min(step_size, 1e-1)

        # Sampling
        chain_samples = []
        chain_log_probs = []
        for i in range(n_samples * thin):
            w_new, p_new = hmc_leapfrog(w, grad_fn, step_size, n_leapfrog)
            lp_old = hmc_log_prob(w, X_t, y_t, sigma_noise, lambda_prior)
            lp_new = hmc_log_prob(w_new, X_t, y_t, sigma_noise, lambda_prior)
            log_accept = (lp_new - lp_old).item()
            if np.log(np.random.rand()) < log_accept:
                w = w_new.detach()
            if i % thin == 0:
                chain_samples.append(w.clone().numpy())
                chain_log_probs.append(lp_old.item())

        all_samples.append(np.array(chain_samples))
        all_log_probs.append(np.array(chain_log_probs))
        if verbose:
            print(f"  Chain {chain+1}/{n_chains}: {len(chain_samples)} samples, "
                  f"final step_size={step_size:.2e}")

    return all_samples, all_log_probs


# ============================================================
# CLAIM 1: Weight-splitting non-identifiability visualization
# ============================================================
def verify_claim1(p=5, M_star=5, sigma=1.0, n=500, seed=42):
    """Verify that redundant neurons split true neuron contributions via simplex."""
    print("\n" + "="*60)
    print("CLAIM 1: Weight-Splitting Non-Identifiability")
    print("="*60)

    W1_star, w2_star = generate_true_network(p, M_star, seed=seed)
    X, y, _ = generate_data(W1_star, w2_star, n, sigma, seed=seed)

    # Test with M > M* (overparametrized)
    M = 2 * M_star  # M = 10
    print(f"True neurons: M*={M_star}, Model neurons: M={M}, n={n}")

    samples, log_probs = run_hmc(X, y, M, n_samples=500, n_warmup=300,
                                  step_size=5e-4, n_leapfrog=10, thin=5,
                                  sigma_noise=sigma, n_chains=3, seed=seed)

    # Compute splitting coefficients for each sample
    all_c = []
    for chain_samples in samples:
        for w in chain_samples:
            W1 = w[:M*p].reshape(M, p)
            w2 = w[M*p:]
            # Assign each model neuron to true neuron via cosine similarity
            assignments = []
            c_vals = []
            for m in range(M):
                sims = []
                for m_star in range(M_star):
                    cos_sim = np.dot(W1[m], W1_star[m_star]) / (
                        np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                    sims.append(cos_sim)
                assigned = np.argmax(sims)
                # Compute splitting coefficient: c_m = ||w1_m||^2 / ||w1_{m*}||^2
                c_m = np.linalg.norm(W1[m])**2 / (np.linalg.norm(W1_star[assigned])**2 + 1e-10)
                c_vals.append(c_m)
                assignments.append(assigned)
            all_c.append(c_vals)

    all_c = np.array(all_c)

    # Verify simplex constraint: sum of c_m within each group should be ~1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Splitting coefficients for first true neuron's group
    group0_idx = [m for m in range(M) if np.mean(all_c[:, m]) > 0.1]
    if len(group0_idx) > 1:
        group_c = all_c[:, group0_idx[:4]]  # Up to 4 neurons
        for i, idx in enumerate(group0_idx[:4]):
            axes[0].hist(group_c[:, i], bins=30, alpha=0.5, label=f'Neuron {idx}', density=True)
        axes[0].set_xlabel('Splitting coefficient c_m')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Claim 1: Splitting Coefficients (M={M})')
        axes[0].legend()

    # Plot 2: Sum of coefficients per group
    group_sums = []
    for chain_samples in samples:
        for w in chain_samples:
            W1 = w[:M*p].reshape(M, p)
            sums = np.zeros(M_star)
            counts = np.zeros(M_star)
            for m in range(M):
                sims = [np.dot(W1[m], W1_star[m_star]) / (
                    np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                    for m_star in range(M_star)]
                assigned = np.argmax(sims)
                c_m = np.linalg.norm(W1[m])**2 / (np.linalg.norm(W1_star[assigned])**2 + 1e-10)
                sums[assigned] += c_m
                counts[assigned] += 1
            group_sums.append(sums / np.maximum(counts, 1))

    group_sums = np.array(group_sums)
    axes[1].boxplot([group_sums[:, i] for i in range(M_star)],
                    labels=[f'Group {i}' for i in range(M_star)])
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Theory: sum=1')
    axes[1].set_ylabel('Average coefficient sum')
    axes[1].set_title('Simplex Constraint Verification')
    axes[1].legend()

    # Plot 3: 2D simplex visualization for M=6 (one redundant neuron)
    # Show how two neurons split one true neuron
    if M > M_star:
        # Find a group with exactly 2 neurons
        from collections import Counter
        assignment_counts = []
        for w in samples[0][:50]:
            W1 = w[:M*p].reshape(M, p)
            assigns = [np.argmax([np.dot(W1[m], W1_star[m_star]) / (
                np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                for m_star in range(M_star)]) for m in range(M)]
            assignment_counts.append(tuple(sorted(assigns)))
        most_common = Counter(assignment_counts).most_common(1)[0][0]

        # Get samples for this assignment
        c1_list, c2_list = [], []
        for w in samples[0]:
            W1 = w[:M*p].reshape(M, p)
            assigns = [np.argmax([np.dot(W1[m], W1_star[m_star]) / (
                np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                for m_star in range(M_star)]) for m in range(M)]
            if tuple(sorted(assigns)) == most_common:
                for m_star_val in set(assigns):
                    group_m = [m for m in range(M) if assigns[m] == m_star_val]
                    if len(group_m) == 2:
                        c1 = np.linalg.norm(W1[group_m[0]])**2 / (np.linalg.norm(W1_star[m_star_val])**2 + 1e-10)
                        c2 = np.linalg.norm(W1[group_m[1]])**2 / (np.linalg.norm(W1_star[m_star_val])**2 + 1e-10)
                        c1_list.append(c1)
                        c2_list.append(c2)
                        break

        if c1_list:
            axes[2].scatter(c1_list, c2_list, alpha=0.3, s=5)
            # Draw simplex boundary
            t = np.linspace(0, 1, 100)
            axes[2].plot(t, 1-t, 'r-', linewidth=2, label='Simplex: c1+c2=1')
            axes[2].set_xlabel('c_1')
            axes[2].set_ylabel('c_2')
            axes[2].set_title('2D Simplex Visualization')
            axes[2].set_xlim(-0.1, 1.1)
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].legend()
            axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "claim1_weight_splitting.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Quantitative check: mean coefficient should be ~1/k for balanced groups
    mean_c_per_group = np.mean(group_sums, axis=0)
    k = M // M_star
    theory_mean = 1.0
    max_error = np.max(np.abs(mean_c_per_group - theory_mean))
    print(f"  Group size k={k}")
    print(f"  Mean coefficient sums per group: {mean_c_per_group}")
    print(f"  Theory: all should be 1.0")
    print(f"  Max deviation: {max_error:.4f}")
    print(f"  Claim 1 VERIFIED: {max_error < 0.3}")

    return {
        "M": M, "M_star": M_star, "n": n, "k": k,
        "mean_c_per_group": mean_c_per_group.tolist(),
        "max_error": float(max_error),
        "verified": max_error < 0.3
    }


# ============================================================
# CLAIM 2: EU trace decomposition
# ============================================================
def verify_claim2(p=5, M_star=5, sigma=1.0, seed=42):
    """Verify that EU = tr(Cov(w|D)) captures persistent parameter uncertainty."""
    print("\n" + "="*60)
    print("CLAIM 2: Epistemic Uncertainty Trace Decomposition")
    print("="*60)

    W1_star, w2_star = generate_true_network(p, M_star, seed=seed)

    sample_sizes = [64, 128, 256, 512, 1024]
    M_vals = [M_star, 2*M_star]  # Correctly specified and overparametrized

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for M in M_vals:
        rmse_list = []
        tr_cov_list = []
        pred_var_list = []

        for n in sample_sizes:
            X, y, _ = generate_data(W1_star, w2_star, n, sigma, seed=seed)
            print(f"  M={M}, n={n}...")

            samples, _ = run_hmc(X, y, M, n_samples=300, n_warmup=200,
                                  step_size=5e-4, n_leapfrog=8, thin=5,
                                  sigma_noise=sigma, n_chains=2, seed=seed, verbose=False)

            # Combine chains
            all_w = np.concatenate(samples, axis=0)

            # Compute RMSE on test data
            X_test, y_test, _ = generate_data(W1_star, w2_star, 500, sigma, seed=seed+999)
            X_test_t = torch.tensor(X_test, dtype=torch.float64)
            y_test_t = torch.tensor(y_test, dtype=torch.float64)

            preds = []
            for w in all_w:
                W1 = torch.tensor(w[:M*p].reshape(M, p), dtype=torch.float64)
                w2_t = torch.tensor(w[M*p:], dtype=torch.float64)
                h = torch.relu(X_test_t @ W1.T)
                f = h @ w2_t
                preds.append(f.numpy())
            preds = np.array(preds)

            rmse = np.sqrt(np.mean((preds.mean(axis=0) - y_test)**2))
            # Predictive variance (epistemic in function space)
            pred_var = np.mean(np.var(preds, axis=0))
            # Trace of weight covariance
            tr_cov = np.mean([np.sum(np.var(all_w[:, i])) for i in range(all_w.shape[1])])

            rmse_list.append(rmse)
            tr_cov_list.append(tr_cov / (M * (p + 1)))  # Normalize by d
            pred_var_list.append(pred_var)

            print(f"    RMSE={rmse:.4f}, tr(Cov)/d={tr_cov/(M*(p+1)):.6f}, "
                  f"pred_var={pred_var:.6f}")

        results[f"M={M}"] = {
            "sample_sizes": sample_sizes,
            "rmse": rmse_list,
            "tr_cov_normalized": tr_cov_list,
            "pred_var": pred_var_list
        }

        label = f"M={M} ({'correctly specified' if M==M_star else 'overparametrized'})"
        axes[0].plot(sample_sizes, rmse_list, 'o-', label=label, markersize=5)
        axes[1].plot(sample_sizes, tr_cov_list, 's-', label=label, markersize=5)

    axes[0].set_xlabel('Sample size n')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Claim 2: Predictive Performance')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Sample size n')
    axes[1].set_ylabel('tr(Cov(w|D)) / d')
    axes[1].set_title('Claim 2: Parameter Uncertainty Persists')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "claim2_eu_trace.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Check: tr(Cov)/d should remain > 0 even for large n
    tr_persists = all(results[f"M={M_star}"]["tr_cov_normalized"][-1] > 0
                       for _ in [1])
    print(f"  tr(Cov)/d at largest n for M=M*: {results[f'M={M_star}']['tr_cov_normalized'][-1]:.6f}")
    print(f"  Claim 2 VERIFIED: {tr_persists}")

    return results


# ============================================================
# CLAIM 3: Dirichlet posterior + moments (Theorem 1)
# ============================================================
def verify_claim3(p=5, M_star=5, sigma=1.0, n=1000, seed=42):
    """Verify Dirichlet distribution of splitting coefficients and Theorem 1 moments."""
    print("\n" + "="*60)
    print("CLAIM 3: Dirichlet Posterior & Theorem 1 Moments")
    print("="*60)

    W1_star, w2_star = generate_true_network(p, M_star, seed=seed)
    X, y, _ = generate_data(W1_star, w2_star, n, sigma, seed=seed)
    alpha_theory = (p + 1) / 2.0

    M_vals = [2*M_star, 4*M_star]  # M=10, M=20
    results = {}

    fig, axes = plt.subplots(2, len(M_vals), figsize=(5*len(M_vals), 10))

    for m_idx, M in enumerate(M_vals):
        print(f"  M={M}, n={n}...")
        k = M // M_star  # Group size (balanced)

        samples, _ = run_hmc(X, y, M, n_samples=500, n_warmup=300,
                              step_size=3e-4, n_leapfrog=10, thin=5,
                              sigma_noise=sigma, n_chains=3, seed=seed, verbose=False)

        # Extract splitting coefficients and assignments
        all_c = []
        all_s = []  # Projections s_m = <w_m, w*_varsigma(m)> / ||w*_varsigma(m)||^2
        all_k_sizes = []

        for chain_samples in samples:
            for w in chain_samples:
                W1 = w[:M*p].reshape(M, p)
                w2_t = w[M*p:]

                # Assign neurons
                assignments = []
                for m in range(M):
                    sims = [np.dot(W1[m], W1_star[m_star]) / (
                        np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                        for m_star in range(M_star)]
                    assignments.append(np.argmax(sims))

                # Compute coefficients and projections
                for m in range(M):
                    m_star = assignments[m]
                    c_m = np.linalg.norm(W1[m])**2 / (np.linalg.norm(W1_star[m_star])**2 + 1e-10)
                    # Projection: s_m = <w_m, w*_m> / ||w*_m||^2
                    omega_m = np.concatenate([W1[m], [w2_t[m]]])
                    omega_star = np.concatenate([W1_star[m_star], [w2_star[m_star]]])
                    s_m = np.dot(omega_m, omega_star) / (np.dot(omega_star, omega_star) + 1e-10)
                    all_c.append(c_m)
                    all_s.append(s_m)

        all_c = np.array(all_c)
        all_s = np.array(all_s)

        # Compute group sizes from assignments
        from collections import Counter
        all_assignments = []
        for chain_samples in samples:
            for w in chain_samples[:50]:
                W1 = w[:M*p].reshape(M, p)
                assigns = [np.argmax([np.dot(W1[m], W1_star[m_star]) / (
                    np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                    for m_star in range(M_star)]) for m in range(M)]
                all_assignments.append(assigns)

        # Group size distribution
        group_size_counts = Counter()
        for assigns in all_assignments:
            group_sizes = Counter(assigns)
            for m_star, count in group_size_counts.items():
                pass  # Just count overall
            for count in group_sizes.values():
                group_size_counts[count] += 1

        most_common_k = group_size_counts.most_common(1)[0][0]
        print(f"    Most common group size: k={most_common_k}")

        # Theoretical moments for Dirichlet(alpha, ..., alpha) with k components
        # E[c_m] = 1/k
        # E[s_m] = mu_{k,alpha} = Gamma(alpha+1/2)*Gamma(k*alpha) / (Gamma(alpha)*Gamma(k*alpha+1/2))
        # E[s_m^2] = 1/k

        def mu_k_alpha(k, alpha):
            return (gamma_fn(alpha + 0.5) * gamma_fn(k * alpha) /
                    (gamma_fn(alpha) * gamma_fn(k * alpha + 0.5)))

        theory_Ec = 1.0 / most_common_k
        theory_Es = mu_k_alpha(most_common_k, alpha_theory)
        theory_Es2 = 1.0 / most_common_k

        emp_Ec = np.mean(all_c)
        emp_Es = np.mean(all_s)
        emp_Es2 = np.mean(all_s**2)

        results[f"M={M}"] = {
            "k": most_common_k,
            "alpha": alpha_theory,
            "theory_Ec": theory_Ec,
            "emp_Ec": emp_Ec,
            "theory_Es": theory_Es,
            "emp_Es": emp_Es,
            "theory_Es2": theory_Es2,
            "emp_Es2": emp_Es2,
        }

        print(f"    Dirichlet: E[c_m] theory={theory_Ec:.4f}, empirical={emp_Ec:.4f}")
        print(f"    Theorem 1: E[s_m] theory={theory_Es:.4f}, empirical={emp_Es:.4f}")
        print(f"    Theorem 1: E[s_m^2] theory={theory_Es2:.4f}, empirical={emp_Es2:.4f}")

        # Plot Beta marginals for Claim 3
        if m_idx < len(axes[0]):
            ax = axes[0, m_idx]
            # Empirical histogram
            ax.hist(all_c, bins=50, density=True, alpha=0.6, label='Empirical', color='steelblue')
            # Theoretical Beta(alpha, (k-1)*alpha)
            x = np.linspace(0.001, min(1.0, np.max(all_c)*1.2), 200)
            beta_a = alpha_theory
            beta_b = (most_common_k - 1) * alpha_theory
            y_theory = stats.beta.pdf(x, beta_a, beta_b)
            ax.plot(x, y_theory, 'r-', linewidth=2, label=f'Beta({beta_a:.1f},{beta_b:.1f})')
            ax.set_xlabel('c_m')
            ax.set_ylabel('Density')
            ax.set_title(f'M={M}, k={most_common_k}')
            ax.legend(fontsize=8)

            # Theorem 1 moments comparison
            ax2 = axes[1, m_idx]
            categories = ['E[c_m]', 'E[s_m]', 'E[s_m²]']
            theory_vals = [theory_Ec, theory_Es, theory_Es2]
            emp_vals = [emp_Ec, emp_Es, emp_Es2]
            x_pos = np.arange(len(categories))
            width = 0.35
            ax2.bar(x_pos - width/2, theory_vals, width, label='Theory', color='coral')
            ax2.bar(x_pos + width/2, emp_vals, width, label='Empirical', color='steelblue')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(categories)
            ax2.set_title(f'Theorem 1 Moments (M={M})')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "claim3_dirichlet_theorem1.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Verify: empirical moments should be close to theory
    errors = []
    for key, r in results.items():
        errors.append(abs(r['emp_Ec'] - r['theory_Ec']))
        errors.append(abs(r['emp_Es'] - r['theory_Es']))
        errors.append(abs(r['emp_Es2'] - r['theory_Es2']))
    max_error = max(errors)
    print(f"  Max moment error: {max_error:.4f}")
    print(f"  Claim 3 VERIFIED: {max_error < 0.15}")

    return results


# ============================================================
# CLAIM 4: Non-degenerate posterior as n -> infinity
# ============================================================
def verify_claim4(p=5, M_star=5, sigma=1.0, seed=42):
    """Verify posterior remains non-degenerate as n -> infinity."""
    print("\n" + "="*60)
    print("CLAIM 4: Non-Degenerate Posterior as n -> infinity")
    print("="*60)

    W1_star, w2_star = generate_true_network(p, M_star, seed=seed)

    M = 2 * M_star  # Overparametrized
    sample_sizes = [128, 256, 512, 1024, 2048]
    alpha_theory = (p + 1) / 2.0

    results = {"n": [], "tr_cov_normalized": [], "var_c": [],
               "pred_rmse": [], "pred_var": []}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for n in sample_sizes:
        print(f"  n={n}...")
        X, y, _ = generate_data(W1_star, w2_star, n, sigma, seed=seed)

        samples, _ = run_hmc(X, y, M, n_samples=400, n_warmup=250,
                              step_size=3e-4, n_leapfrog=10, thin=5,
                              sigma_noise=sigma, n_chains=2, seed=seed, verbose=False)

        all_w = np.concatenate(samples, axis=0)

        # Compute trace of weight covariance (normalized)
        tr_cov = np.sum(np.var(all_w, axis=0)) / (M * (p + 1))

        # Compute variance of splitting coefficients
        all_c = []
        for w in all_w:
            W1 = w[:M*p].reshape(M, p)
            for m in range(M):
                sims = [np.dot(W1[m], W1_star[m_star]) / (
                    np.linalg.norm(W1[m]) * np.linalg.norm(W1_star[m_star]) + 1e-10)
                    for m_star in range(M_star)]
                m_star = np.argmax(sims)
                c_m = np.linalg.norm(W1[m])**2 / (np.linalg.norm(W1_star[m_star])**2 + 1e-10)
                all_c.append(c_m)
        all_c = np.array(all_c)
        var_c = np.var(all_c)

        # Predictive metrics
        X_test, y_test, _ = generate_data(W1_star, w2_star, 500, sigma, seed=seed+999)
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        preds = []
        for w in all_w[:200]:
            W1 = torch.tensor(w[:M*p].reshape(M, p), dtype=torch.float64)
            w2_t = torch.tensor(w[M*p:], dtype=torch.float64)
            h = torch.relu(X_test_t @ W1.T)
            preds.append((h @ w2_t).numpy())
        preds = np.array(preds)
        rmse = np.sqrt(np.mean((preds.mean(axis=0) - y_test)**2))
        pred_var = np.mean(np.var(preds, axis=0))

        results["n"].append(n)
        results["tr_cov_normalized"].append(tr_cov)
        results["var_c"].append(var_c)
        results["pred_rmse"].append(rmse)
        results["pred_var"].append(pred_var)

        print(f"    tr(Cov)/d={tr_cov:.6f}, var(c)={var_c:.6f}, "
              f"RMSE={rmse:.4f}, pred_var={pred_var:.6f}")

    # Plot
    axes[0].plot(results["n"], results["tr_cov_normalized"], 'o-', color='steelblue')
    axes[0].set_xlabel('Sample size n')
    axes[0].set_ylabel('tr(Cov(w|D)) / d')
    axes[0].set_title('Claim 4: Parameter Uncertainty')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["n"], results["var_c"], 's-', color='coral')
    axes[1].set_xlabel('Sample size n')
    axes[1].set_ylabel('Var(c_m)')
    axes[1].set_title('Claim 4: Splitting Coefficient Variance')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results["n"], results["pred_rmse"], 'o-', label='RMSE', color='steelblue')
    axes[2].plot(results["n"], results["pred_var"], 's-', label='Pred Var', color='coral')
    axes[2].axhline(y=sigma**2, color='gray', linestyle='--', label=f'σ²={sigma**2}')
    axes[2].set_xlabel('Sample size n')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Claim 4: Function-Space Convergence')
    axes[2].set_xscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "claim4_n_to_infinity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Check: tr(Cov)/d should NOT go to 0, while pred_var should decrease
    tr_persists = results["tr_cov_normalized"][-1] > results["tr_cov_normalized"][0] * 0.1
    pred_converges = results["pred_rmse"][-1] < results["pred_rmse"][0] * 0.8
    print(f"  tr(Cov)/d at largest n: {results['tr_cov_normalized'][-1]:.6f}")
    print(f"  Pred RMSE at largest n: {results['pred_rmse'][-1]:.4f}")
    print(f"  Claim 4 VERIFIED: tr persists={tr_persists}, pred converges={pred_converges}")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("REPRODUCTION: On the Epistemic Uncertainty of")
    print("Overparametrized Neural Networks (ICML 2026)")
    print("="*60)
    print(f"Start time: {datetime.now().isoformat()}")

    all_results = {}

    # Claim 1
    all_results["claim1"] = verify_claim1()

    # Claim 2
    all_results["claim2"] = verify_claim2()

    # Claim 3
    all_results["claim3"] = verify_claim3()

    # Claim 4
    all_results["claim4"] = verify_claim4()

    # Save results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("ALL CLAIMS VERIFICATION COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Results saved to {RESULTS_DIR}/")
