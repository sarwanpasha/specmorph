# Spectral Morphogenesis

This repository contains the reference implementation and experiments for **Spectral Morphogenesis**, a framework that studies catastrophic
forgetting in continual learning through the lens of *representation coherence* and
sheaf-theoretic perturbation theory.

The central object is a **coherence functional** measured on the learned
representation. We derive a perturbation bound that predicts how much a previously
learned task degrades when a network is updated on new data, and we show that the
bound is *quantitatively binding* in pretrained-backbone (frozen-feature) settings.
Coherence is used primarily as a **diagnostic instrument** for continual learning
rather than as a mechanism claimed to outperform strong projection baselines.

---

## Key ideas

- **Coherence functional.** A scalar summarising the mutual alignment of the
  subspaces a model uses across tasks. Low coherence corresponds to interference
  that is easier to control.

- **Perturbation bound.** A closed-form upper bound on relative forgetting of an
  earlier task expressed in terms of a spectral quantity of the update and the
  aggregate coherence of the representation. The bound is derived, re-derived, and
  numerically verified rather than asserted.

- **Binding regime.** On a frozen pretrained vision backbone (ViT), the measured
  spectral quantity, the aggregate coherence, and the resulting bound ceiling are
  all sub-maximal, and the empirically measured forgetting respects the predicted
  ceiling. This is the regime where the theory is tightest.

- **Comparison to GPM.** Across the regimes we can measure (permuted-MNIST,
  split-CIFAR, TinyImageNet, prompt-based baselines), the coherence-based method
  **matches** Gradient Projection Memory (GPM) but does not beat it. We report this
  directly; coherence's value here is as a measurement and diagnostic tool.

---

## Repository layout

```
.
|-- src/
|   |-- theory/        # Numerical verification of the theoretical results
|   |   |-- verify_theory_indep.py      # independence / core bound checks
|   |   |-- verify_theory_indep_B.py    # variant / robustness checks
|   |   |-- verify_obstruction.py       # obstruction-term verification
|   |   |-- verify_lowerbound.py        # lower-bound / tightness checks
|   |   |-- verify_gram.py              # Gram-matrix / coherence identities
|   |
|   |-- method/        # Core coherence-based method implementations
|   |   |-- spectral_morph_v3.py        # main method (latest)
|   |   |-- spectral_morph_v3b.py       # ablation variant
|   |   |-- spectral_morph_v2.py        # earlier iteration (kept for reference)
|   |   |-- spectral_morph_test.py      # unit-style sanity checks
|   |   |-- cobound_pmnist.py           # coherence-bound method, permuted-MNIST
|   |   |-- cobound_pmnist2.py          # delta-aware variant
|   |   |-- cobound_cifar.py            # coherence-bound method, split-CIFAR
|   |   |-- cobound_cifar2.py           # delta-aware variant
|   |
|   |-- experiments/   # End-to-end continual-learning experiments
|   |   |-- realgraph_pmnist.py         # permuted-MNIST, GPM vs coherence
|   |   |-- realgraph_cifar.py          # split-CIFAR, GPM vs coherence
|   |   |-- realgraph_*_pretrgp.py      # pretrained-backbone GPM variants
|   |   |-- realgraph_pmnist_prederpp.py# ER++ baseline comparison
|   |   |-- vision_pmnist.py            # vision pipeline, permuted-MNIST
|   |   |-- vision_pmnist_mu.py         # + coherence measurement
|   |   |-- vision_cifar_resnet.py      # ResNet split-CIFAR
|   |   |-- vision_cifar_rn_diag.py     # + diagnostics
|   |   |-- vision_ci.py / vision_ci2.py# class-incremental settings
|   |   |-- vision_ms.py                # multi-seed driver
|   |   |-- tin_cl_projection.py        # TinyImageNet, projection method
|   |   |-- tin_l2p.py                  # TinyImageNet, L2P prompt baseline
|   |   |-- extract_tinyimagenet_feats.py # frozen-backbone feature extraction
|   |   |-- eff_exp.py                  # memory/compute efficiency study
|   |   |-- taskfree.py / taskfree2.py  # task-free / online streaming study
|   |   |-- b1_ablation.py              # ablation
|   |
|   |-- probes/        # Small measurement / analysis probes
|       |-- mu_frame_probe.py           # coherence (mu) framewise probe
|       |-- sintheta_probe.py           # principal-angle (sin theta) probe
|       |-- gram_real.py                # Gram-matrix measurement on real data
|       |-- binding_bound.py            # binding-regime bound evaluation
|       |-- frozen_proj.py              # frozen-backbone projected update
|       |-- frozen_bound.py             # frozen-backbone bound check
|
|-- results/                           # (populated when you run experiments)
|-- run_example.sbatch                 # generic SLURM template
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Installation

Requires Python 3.9+ and a working PyTorch install (GPU recommended for the
vision and TinyImageNet experiments; the MNIST-scale scripts run on CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
scipy>=1.10
```

---

## Data

- **MNIST / permuted-MNIST** and **CIFAR** are downloaded automatically by
  `torchvision` into a local `./data` (or `./MNIST`) directory on first run.
- **TinyImageNet** must be downloaded separately and placed under `./data`;
  `extract_tinyimagenet_feats.py` computes frozen-backbone features used by the
  TinyImageNet experiments.

Downloaded data and caches are git-ignored.

---

## Running

Each script is self-contained and prints machine-readable `RESULT ...` lines to
stdout. For a quick local check:

```bash
python src/experiments/realgraph_pmnist.py
python src/theory/verify_theory_indep.py
python src/probes/mu_frame_probe.py
```

On an HPC cluster, adapt the generic template and submit:

```bash
# edit run_example.sbatch: set <ACCOUNT>, <PARTITION>, and PYTHON
sbatch run_example.sbatch
```

---

## Reproducing the main results

| Result | Script | What it shows |
|--------|--------|---------------|
| Core bound (numeric) | `src/theory/verify_theory_indep.py` | Re-derived bound holds on synthetic perturbations |
| Coherence identities | `src/theory/verify_gram.py` | Gram/coherence measurements are internally consistent |
| Binding regime | `src/probes/frozen_proj.py`, `src/probes/binding_bound.py` | On a frozen ViT backbone, measured forgetting respects the predicted ceiling |
| permuted-MNIST | `src/experiments/realgraph_pmnist.py` | Coherence method matches GPM |
| split-CIFAR | `src/experiments/realgraph_cifar.py` | Coherence method matches GPM |
| TinyImageNet | `src/experiments/tin_cl_projection.py`, `tin_l2p.py` | Scaling to a larger benchmark; L2P prompt baseline |
| Efficiency study | `src/experiments/eff_exp.py` | Memory/compute trade-off vs a growing-basis baseline |
| Task-free / online | `src/experiments/taskfree.py`, `taskfree2.py` | Streaming regime with no task boundaries |

---

## License

Released under the MIT License. See `LICENSE` (add before publishing).
