# 64x8 Massive MIMO Uplink Simulation

This repository contains three versions of the 64-antenna, 8-user uplink study.

## Version 1
MassiveMIMO64x8.ipynb — QPSK, i.i.d. Rayleigh, artificial imperfect CSI,
MRC/ZF/MMSE detection, BER and simplified SE/EE proxies.

## Version 2
Massive_MIMO_64x8_Version2.ipynb — QPSK/16-QAM/64-QAM, large-scale fading,
spatially correlated ULA, multipath OFDM, AR(1)-approximated Doppler,
orthogonal-pilot LS/LMMSE, MRC/ZF/MMSE, actual post-detection SINR and
illustrative power modelling.

## Version 3
Massive_MIMO_64x8_Version3.ipynb — modular Python implementation under version3/.
Adds multi-seed evaluation, CI95 summaries, receiver-specific SINR, tests and
reproducible CSV/plot outputs.

Run V3:
    cd version3
    python -m pip install -r requirements.txt
    python run_experiment.py --seeds 5 --blocks 50
    pytest -q

Modelling caveats: orthogonal pilots are assumed; Doppler and CP overhead are
approximations; SE/EE use proxies and an illustrative power model. See
version3/README.md and version3/VALIDATION_REPORT.md.
