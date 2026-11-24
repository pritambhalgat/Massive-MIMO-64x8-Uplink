# Massive MIMO 64x8 — Version 3

A modular, reproducible implementation of the Version 2 link model. It adds
multi-seed evaluation, confidence intervals, testable Python modules and
receiver-specific post-combining SINR.

## Model
M=64 antennas, K=8 uplink users, 8 OFDM subcarriers, 4 multipath taps;
QPSK/16-QAM/64-QAM; large-scale fading, correlated ULA and AR(1)-approximated
Doppler; orthogonal-pilot LS/LMMSE; MRC/ZF/MMSE.

For combiner W and G=WH:
SINR_k = |G_kk|^2 / (sum_{j!=k}|G_kj|^2 + sigma2*||w_k||^2).
The normalized equal-symbol-power link model is used after transmit-power normalization.

## Run
    python -m pip install -r requirements.txt
    python run_experiment.py --seeds 5 --blocks 50
    pytest -q

## Outputs
results_v3/mimo_v3_per_seed.csv
results_v3/mimo_v3_summary_ci95.csv
one BER/SE/EE plot per modulation.

## Assumptions and limitations
Orthogonal pilots are assumed; pilot contamination is omitted. Doppler uses
a first-order Gauss-Markov approximation. CP and pilot overhead are proxies.
SE is a modulation-capped SINR proxy, not coded goodput. The detector power
model is illustrative and needs calibration before energy-efficiency claims.
No HARQ, RF impairments, scheduling or full 3GPP compliance is claimed.
95% confidence intervals are across independent seeds; use more than five
seeds and more blocks for final research analysis.