#!/usr/bin/env bash
set -euo pipefail

mkdir -p paper_v3/figures

python make_fig1.py
python make_fig2_full.py
python run_fig3_k_sweep.py --plot_only
python plot_fig4_final.py
python run_fig5_corr_sweep.py --plot_only
python plot_v2_n1300_pct_supp.py
python plot_v2_n1300_angle_hist.py
python make_supp_fig_metrics.py --merge
python plot_learning_rule_combs_bopt1.py
python plot_highcorr_covariance_rule.py
