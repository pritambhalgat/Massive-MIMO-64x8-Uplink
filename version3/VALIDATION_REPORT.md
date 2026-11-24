# Version 3 validation report

- Python syntax compilation: passed.
- Unit tests: 3 passed.
- One-SNR QPSK smoke run completed for LS/LMMSE and MRC/ZF/MMSE.
- A broader smoke run completed with 2 seeds x 2 blocks, all 7 SNRs,
  3 modulation orders, 2 estimators and 3 detectors. Per-seed and 95% CI CSVs
  and plots were generated.

This is software smoke validation only, not publication-grade evidence.
Two seeds and two blocks are inadequate for reliable performance claims.
Increase the experiment to 10-20 seeds and 100-500 blocks, and report BER,
SE proxy, EE proxy and their confidence intervals together.

The power model and pilot/CP overhead remain illustrative. No coded goodput,
pilot contamination, HARQ, RF impairment or 3GPP compliance is claimed.