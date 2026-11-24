import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from src.simulator import simulate_seed, summarize_runs

def main():
    parser = argparse.ArgumentParser(description="Run reproducible 64x8 Massive MIMO Version 3.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--output", type=str, default="results_v3")
    args = parser.parse_args()
    if args.seeds < 1 or args.blocks < 1:
        raise ValueError("--seeds and --blocks must be positive.")
    cfg = Config(n_blocks=args.blocks)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(args.seed_start, args.seed_start+args.seeds):
        print(f"Running seed {seed} ({args.blocks} blocks/SNR)...")
        runs.append(simulate_seed(seed, cfg, args.blocks))
    raw = pd.concat(runs, ignore_index=True)
    summary = summarize_runs(raw)
    raw.to_csv(out/"mimo_v3_per_seed.csv", index=False)
    summary.to_csv(out/"mimo_v3_summary_ci95.csv", index=False)
    for modulation in summary["modulation"].unique():
        subset = summary[(summary["modulation"]==modulation) & (summary["estimator"]=="LMMSE")]
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        for detector in cfg.detectors:
            g = subset[subset["detector"]==detector].sort_values("snr_db")
            axes[0].semilogy(g["snr_db"], g["ber_mean"].clip(lower=1e-7), marker="o", label=detector)
            axes[1].plot(g["snr_db"], g["net_se_proxy_bit_s_hz_mean"], marker="o", label=detector)
            axes[2].plot(g["snr_db"], g["ee_proxy_bit_j_hz_mean"], marker="o", label=detector)
        axes[0].set_ylabel("BER"); axes[0].set_title(f"{modulation} BER")
        axes[1].set_ylabel("bit/s/Hz"); axes[1].set_title("Net SE proxy")
        axes[2].set_ylabel("bit/J/Hz"); axes[2].set_title("EE proxy")
        for ax in axes:
            ax.set_xlabel("SNR (dB)"); ax.grid(True, which="both"); ax.legend()
        fig.tight_layout()
        fig.savefig(out/f"{modulation.replace('-','_')}_v3.png", dpi=180)
        plt.close(fig)
    print(f"Saved {out/'mimo_v3_per_seed.csv'} and {out/'mimo_v3_summary_ci95.csv'}")
if __name__ == "__main__":
    main()
