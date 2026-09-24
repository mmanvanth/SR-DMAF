# SR-DMAF: reproducible experiments

Everything reported in the paper is produced by these scripts.

## Requirements
Python 3.10+, numpy, pandas, scikit-learn, scipy, networkx, matplotlib. No GPU needed; the whole pipeline runs on one CPU core in about 15 minutes.

## Data
Download `ransom.csv` (Ransomware Dataset 2024, https://doi.org/10.5281/zenodo.13890887, Kaggle mirror: mexwell/ransomware-dataset-2024) and set the path in `exp_classify.py` (DATA) and `exp_srvl.py`.

## Run
1. `python exp_classify.py` - removes the 24 label-conflicting hashes, builds the three feature views, trains agents A1-A4 and boosted trees under hash-grouped and leaky splits (5 seeds). Writes `results_classification.csv` and `store.pkl`.
2. `python exp_srvl.py` - calibrates SRVL on validation data, rates every test decision, and runs the 20-topology cascade simulation for all policies and ablations. Writes `results_policies.csv`, `results_reliability.csv`, `results_calibration.csv`, `results_latency.json`.
3. `figures/*.py` - regenerate Figs. 1-4 (fig3 is produced by `fig3_and_legacy.py`; its fig1/fig2/fig4 outputs are superseded by the dedicated scripts).

## Files
- `core.py` - feature views, numpy MC-dropout MLP, topology generator, cascade model (Eq. 1), blast radius (Eq. 4)
- `results/` - the exact result files used in the paper

## Modelling assumptions (see paper, Sect. 5.3 and 6.6)
Efficacy of actions: T4 = 1.0, T1+T2+T3 = e_G (0.95; sensitivity 0.80-1.00), T1 alone = 0.80. Graduated actions keep dependency edges. Cascade threshold theta = 0.5. tau_d = 0.40.
