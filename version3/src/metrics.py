def total_power_w(detector, m, k, cfg):
    extra = cfg.detector_power_map[detector]
    return (cfg.static_w + m*cfg.rf_w_per_antenna
            + m*k*cfg.bb_w_per_mk
            + k*cfg.ue_w_per_user/cfg.pa_efficiency + extra)

def energy_efficiency_proxy(se_bit_s_hz, power_w):
    return float(se_bit_s_hz/max(power_w, 1e-12))
