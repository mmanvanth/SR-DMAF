# SR-DMAF: reproducible experiments

Everything reported in the paper is produced by these scripts.

## Requirements
Python 3.10+, numpy, pandas, scikit-learn, scipy, networkx, matplotlib. No GPU needed; the whole pipeline runs on one CPU core in about 15 minutes.

## Data
Download `ransom.csv` (Ransomware Dataset 2024, https://doi.org/10.5281/zenodo.13890887, Kaggle mirror: mexwell/ransomware-dataset-2024) and set the path in `exp_classify.py` (DATA) and `exp_srvl.py`.

## Run
1. `python exp_classify.py` - removes the 24 label-conflicting hashes, builds the three feature views, trains agents A1-A4 and boosted trees under hash-grouped and leaky splits (5 seeds). Writes `results_classification.csv` and `store.pkl`.
2. `python exp_srvl.py` - calibrates SRVL on validation data, rates every test decision, and runs the 20-topology cascade simulation for all policies and ablations. Writes `results_policies.csv`, `results_reliability.csv`, `results_calibration.csv`, `results_latency.json`.
3. `python exp_sensitivity.py` - reuses the stored test predictions (no retraining) to repeat the containment experiment across 12 network variants (failure threshold, size, Tier-1 count, replicas, dependency density, weight range) and across assumed containment success rates. Writes `results_sensitivity_topology.csv` and `results_sensitivity_efficacy.csv` (Tables 6 and 7 of the paper).
4. `python exp_calibration.py` - computes the expected calibration error, Brier scores and reliability-diagram bins of A4 (single pass vs. Monte Carlo mean) from the stored predictions, and the exact percentage reductions reported in the paper. Writes `results_calibration_ece.csv` and `results_reliability_diagram.json` (Table 4).
5. `python exp_riskscore.py` - ablation of the blast-radius risk score components, fitted versus hand-set weights on held-out networks, and operating metrics at tau_d (Table 7). Writes `results_riskscore.json`.
6. `python exp_stats_side.py` - paired bootstrap significance test (SR-DMAF vs. naive isolation), seed standard deviations for Tables 5, 8 and 9, and the side-effect sensitivity of graduated actions. Writes `results_stats_side.json`.
7. `figures/*.py` - regenerate Figs. 1-4 (fig3 is produced by `fig3_and_legacy.py`; its fig1/fig2/fig4 outputs are superseded by the dedicated scripts).

## Files
- `core.py` - feature views, numpy MC-dropout MLP, topology generator, cascade model (Eq. 1), blast radius (Eq. 4)
- `results/` - the exact result files used in the paper

## Modelling assumptions (see paper, Sect. 5.3; sensitivity in Sect. 6.5)
Efficacy of actions: T4 = 1.0, T1+T2+T3 = e_G (0.95; sensitivity 0.80-1.00), T1 alone = 0.80. Graduated actions keep dependency edges. Cascade threshold theta = 0.5. tau_d = 0.40.
