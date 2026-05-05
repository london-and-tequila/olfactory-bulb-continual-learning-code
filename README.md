# Neurogenesis-Inspired Continual Odor Discrimination

This repository contains a clean code release for the manuscript
**"Neurogenesis-Inspired Continual Discrimination Learning via Sparse Granule-Cell Recruitment."**

The code simulates sequential odor-discrimination tasks in a two-population
olfactory-bulb model and reproduces the main figure panels and supplementary
robustness plots from the paper. Inputs are synthetic and generated on the fly;
no external dataset or pretrained model is required.

## Contents

- `lib/`: model, input generator, and experiment driver.
- `run_neurogenesis_multi_pair.py`: single experiment entry point.
- `run_fig2_baseline.py`, `run_fig3_k_sweep.py`, `run_fig4_cl_sweep.py`,
  `run_fig5_corr_sweep.py`: scripts for the main experiment sweeps.
- `make_fig2_full.py`, `plot_fig4_final.py`, `make_supp_fig_metrics.py`,
  `plot_v2_n1300_pct_supp.py`, `plot_v2_n1300_angle_hist.py`: plotting scripts.
- `data/learning_rule_controls/`: small angle-vs-lag arrays used for the
  learning-rule control supplementary figures.
- `scripts/`: convenience scripts for smoke tests, full sweeps, and plotting.

Generated outputs are written under `results/` and `paper_v3/figures/`.
Those directories are intentionally ignored by git.

## Environment

The full experiments were run with GPU acceleration. A CUDA 12 JAX environment
can be installed with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that JAX sees a GPU before running full sweeps:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

## Quick Smoke Test

This tiny run checks that the code path executes. It is not a paper experiment.

```bash
bash scripts/run_smoke_test.sh
```

## Reproducing the Paper Figures

The complete paper sweeps use 4 seeds (`0,1,2,3`), 300 held-out task pairs per
condition, and 1300 sequential training pairs for the v2 allocation rules. To
run all main sweeps from scratch:

```bash
bash scripts/reproduce_main_sweeps.sh
```

After the summaries exist under `results/`, regenerate all current manuscript
figures with:

```bash
bash scripts/plot_all_figures.sh
```

The main generated files are:

- `paper_v3/figures/fig1.pdf`
- `paper_v3/figures/fig2.pdf`
- `paper_v3/figures/fig_k_sweep.pdf`
- `paper_v3/figures/fig_similarity_sweep.pdf`
- `paper_v3/figures/fig_coding_sparsity.pdf`
- supplementary figures in the same directory.

## Notes

- The experiments use synthetic Gaussian-copula input patterns, described in
  the manuscript appendix.
- The main figures pool 4 seeds and 300 test memories unless otherwise noted.
- The high-correlation covariance-rule supplementary figure is a targeted
  single-seed control and is not a full allocation-strategy comparison.
- For double-anonymous review, use an anonymized repository URL or submit this
  repository as anonymized supplementary material.
