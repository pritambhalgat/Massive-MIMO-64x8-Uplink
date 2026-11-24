import numpy as np

def j0_approx(x: float) -> float:
    term = 1.0
    total = 1.0
    for n in range(1, 30):
        term *= -(x*x/4.0)/(n*n)
        total += term
        if abs(term) < 1e-13:
            break
    return float(total)

def large_scale_fading(k, rng, pathloss_exponent=3.5, shadow_std_db=6.0):
    distance = rng.uniform(50.0, 250.0, k)
    shadow_db = rng.normal(0.0, shadow_std_db, k)
    rel_loss_db = 10.0*pathloss_exponent*np.log10(distance/100.0) + shadow_db
    beta = 10.0**(-rel_loss_db/10.0)
    return distance, shadow_db, beta/np.mean(beta)

def spatial_correlation(m, rho):
    idx = np.arange(m)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    eigval, eigvec = np.linalg.eigh(R)
    R_sqrt = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 1e-12))) @ eigvec.conj().T
    return R, R_sqrt

def doppler_parameters(speed_kmh, carrier_hz, block_seconds):
    speed = speed_kmh/3.6
    fd = speed/(3e8/carrier_hz)
    alpha = float(np.clip(j0_approx(2*np.pi*fd*block_seconds), -0.99, 0.99))
    return fd, alpha

def generate_taps(beta, R_sqrt, power_delay, rng):
    L = len(power_delay)
    m = R_sqrt.shape[0]
    k = len(beta)
    taps = np.empty((L, m, k), dtype=np.complex128)
    for ell in range(L):
        for user in range(k):
            z = (rng.normal(size=m) + 1j*rng.normal(size=m))/np.sqrt(2.0)
            taps[ell, :, user] = np.sqrt(beta[user]*power_delay[ell])*(R_sqrt @ z)
    return taps

def evolve_taps(old_taps, beta, R_sqrt, power_delay, alpha, rng):
    innovation = generate_taps(beta, R_sqrt, power_delay, rng)
    return alpha*old_taps + np.sqrt(max(0.0, 1.0-alpha**2))*innovation

def frequency_response(taps, subcarrier, n_subcarriers):
    phase = np.exp(-1j*2*np.pi*subcarrier*np.arange(taps.shape[0])/n_subcarriers)
    return np.einsum("lmk,l->mk", taps, phase)
