from dataclasses import dataclass

@dataclass
class Config:
    M: int = 64
    K: int = 8
    n_subcarriers: int = 8
    n_taps: int = 4
    coherence_symbols: int = 20
    n_blocks: int = 50
    snr_db: tuple = (0, 5, 10, 15, 20, 25, 30)
    modulations: tuple = (("QPSK", 4), ("16-QAM", 16), ("64-QAM", 64))
    estimators: tuple = ("LS", "LMMSE")
    detectors: tuple = ("MRC", "ZF", "MMSE")
    spatial_rho: float = 0.35
    pathloss_exponent: float = 3.5
    shadow_std_db: float = 6.0
    carrier_hz: float = 3.5e9
    speed_kmh: float = 30.0
    block_seconds: float = 1e-3
    pilot_to_data_power: float = 1.0
    static_w: float = 80.0
    rf_w_per_antenna: float = 0.8
    bb_w_per_mk: float = 0.015
    ue_w_per_user: float = 0.1
    pa_efficiency: float = 0.35
    detector_extra_w: tuple = (("MRC", 0.0), ("ZF", 1.0), ("MMSE", 1.5))

    @property
    def modulation_map(self):
        return dict(self.modulations)

    @property
    def detector_power_map(self):
        return dict(self.detector_extra_w)
