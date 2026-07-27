# Reproducing "On the Epistemic Uncertainty of Overparametrized Neural Networks"

**Paper**: David Rügamer, ICML 2026 — [arXiv:2605.25234](https://arxiv.org/abs/2605.25234)

## Claims Verified

| Claim | Status |
|---|---|
| **Claim 1**: Weight-splitting non-identifiability (simplex constraints) | ✅ Verified |
| **Claim 2**: EU = tr(Cov(w\|D)) captures persistent uncertainty | ✅ Verified |
| **Claim 3**: Dirichlet posterior with α=(p+1)/2 & Theorem 1 moments | ⚠️ Partially verified |
| **Claim 4**: Non-degenerate posterior as n→∞ | ✅ Verified |

## Setup

| Parameter | Value |
|---|---|
| True neurons M* | 5 |
| Inputs p | 5 |
| Noise σ | 1.0 |
| Prior | w ~ N(0, (2λ)⁻¹I), λ=1.0 |
| Sampler | Custom leapfrog HMC |
| Chains | 2-3 per configuration |
| Network widths M | 5, 10, 20 |
| Sample sizes n | 64–512 |

## Quick Start

```bash
pip install torch numpy scipy matplotlib
python repro_code/run_all.py
```

Outputs are saved to `repro_code/results/`:
- `claim1.png` — Splitting coefficient marginals vs Dirichlet theory
- `claim2.png` — Predictive convergence + persistent parameter uncertainty
- `claim3.png` — Beta marginals + Theorem 1 moments
- `claim4.png` — Non-degenerate posterior as n increases

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](repro_code/Reproduction_ICML2026_Epistemic_Uncertainty.ipynb)

## Logbook

Trackio logbook served at `.trackio/logbook/`

## Citation

```bibtex
@inproceedings{rugamer2026epistemic,
  title={On the Epistemic Uncertainty of Overparametrized Neural Networks},
  author={R\"ugamer, David},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
