import numpy as np
import pandas as pd
from config import Config
from src.channel import (large_scale_fading, spatial_correlation, doppler_parameters,
                         generate_taps, evolve_taps, frequency_response)
from src.modem import bits_per_symbol, qam_mod, qam_demod
from src.estimation import make_lmmse_filters, estimate_channel
from src.detectors import make_combiner, post_detection_sinr
from src.metrics import total_power_w, energy_efficiency_proxy

def simulate_seed(seed, cfg=None, n_blocks=None):
    cfg = cfg or Config()
    n_blocks = int(n_blocks or cfg.n_blocks)
    rng = np.random.default_rng(seed)
    M, K = cfg.M, cfg.K
    _, _, beta = large_scale_fading(K, rng, cfg.pathloss_exponent, cfg.shadow_std_db)
    R, R_sqrt = spatial_correlation(M, cfg.spatial_rho)
    power_delay = np.exp(-np.arange(cfg.n_taps, dtype=float))
    power_delay /= power_delay.sum()
    fd_hz, alpha = doppler_parameters(cfg.speed_kmh, cfg.carrier_hz, cfg.block_seconds)
    rows = []
    for snr_db in cfg.snr_db:
        data_power = 10.0**(float(snr_db)/10.0)
        noise_var = 1.0
        noise_eff = noise_var/data_power
        pilot_power = cfg.pilot_to_data_power*data_power
        lmmse_filters = make_lmmse_filters(beta, R, pilot_power, noise_var)
        taps = generate_taps(beta, R_sqrt, power_delay, rng)
        acc = {}
        for mod_name, order in cfg.modulations:
            for estimator in cfg.estimators:
                for detector in cfg.detectors:
                    acc[(mod_name, order, estimator, detector)] = {
                        "errors": 0, "bits": 0, "se_sum": 0.0, "sinr_sum": 0.0, "sinr_n": 0
                    }
        for block in range(n_blocks):
            if block:
                taps = evolve_taps(taps, beta, R_sqrt, power_delay, alpha, rng)
            for sc in range(cfg.n_subcarriers):
                H = frequency_response(taps, sc, cfg.n_subcarriers)
                Hhat = {
                    est: estimate_channel(H, beta, R, pilot_power, noise_var, est, rng, lmmse_filters)
                    for est in cfg.estimators
                }
                combiners = {
                    (est, det): make_combiner(Hhat[est], noise_eff, det)
                    for est in cfg.estimators for det in cfg.detectors
                }
                for mod_name, order in cfg.modulations:
                    bps = bits_per_symbol(order)
                    tx_bits = rng.integers(0, 2, K*bps, dtype=np.int8)
                    x = qam_mod(tx_bits, order)
                    awgn = (rng.normal(size=M)+1j*rng.normal(size=M))*np.sqrt(noise_var/2.0)
                    y = (np.sqrt(data_power)*(H @ x)+awgn)/np.sqrt(data_power)
                    for est in cfg.estimators:
                        for det in cfg.detectors:
                            W = combiners[(est, det)]
                            raw = W @ y
                            gain = np.diag(W @ Hhat[est])
                            safe_gain = np.where(np.abs(gain)>1e-12, gain, 1e-12+0j)
                            rx_bits = qam_demod(raw/safe_gain, order)
                            sinr = np.clip(post_detection_sinr(W, H, noise_eff), 0.0, 1e12)
                            item = acc[(mod_name, order, est, det)]
                            item["errors"] += int(np.sum(tx_bits != rx_bits))
                            item["bits"] += int(tx_bits.size)
                            item["se_sum"] += float(np.sum(np.minimum(np.log2(1.0+sinr), bps)))
                            item["sinr_sum"] += float(np.sum(sinr))
                            item["sinr_n"] += K
        for (mod_name, order, est, det), item in acc.items():
            ber = item["errors"]/max(item["bits"], 1)
            se = item["se_sum"]/(n_blocks*cfg.n_subcarriers)
            se *= max(0.0, 1.0-K/cfg.coherence_symbols)
            se *= max(0.0, 1.0-(cfg.n_taps-1)/cfg.n_subcarriers)
            mean_sinr = item["sinr_sum"]/max(item["sinr_n"], 1)
            power = total_power_w(det, M, K, cfg)
            rows.append({
                "seed": int(seed), "snr_db": float(snr_db), "modulation": mod_name,
                "qam_order": int(order), "estimator": est, "detector": det,
                "blocks": n_blocks, "ber": float(ber), "net_se_proxy_bit_s_hz": float(se),
                "ee_proxy_bit_j_hz": energy_efficiency_proxy(se, power),
                "mean_post_sinr_db": float(10*np.log10(max(mean_sinr, 1e-15))),
                "power_model_w": float(power), "doppler_hz": float(fd_hz),
                "doppler_ar1_alpha": float(alpha),
                "pilot_overhead_factor": float(max(0.0,1.0-K/cfg.coherence_symbols)),
                "cp_overhead_factor": float(max(0.0,1.0-(cfg.n_taps-1)/cfg.n_subcarriers))
            })
    return pd.DataFrame(rows)

def summarize_runs(raw_df):
    from scipy.stats import t
    metrics = ["ber", "net_se_proxy_bit_s_hz", "ee_proxy_bit_j_hz", "mean_post_sinr_db"]
    keys = ["snr_db", "modulation", "estimator", "detector"]
    rows = []
    for key, group in raw_df.groupby(keys):
        row = dict(zip(keys, key))
        seed_count = group["seed"].nunique()
        for metric in metrics:
            values = group.groupby("seed")[metric].mean().to_numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values)>1 else float("nan")
            half = float(t.ppf(0.975, len(values)-1)*std/np.sqrt(len(values))) if len(values)>1 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std_across_seeds"] = std
            row[f"{metric}_ci95_low"] = mean-half if np.isfinite(half) else float("nan")
            row[f"{metric}_ci95_high"] = mean+half if np.isfinite(half) else float("nan")
        row["n_independent_seeds"] = seed_count
        rows.append(row)
    return pd.DataFrame(rows)
