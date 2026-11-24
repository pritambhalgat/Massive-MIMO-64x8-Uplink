import numpy as np

def make_lmmse_filters(beta, spatial_covariance, pilot_power, noise_var):
    m = spatial_covariance.shape[0]
    identity = np.eye(m)
    filters = []
    for beta_k in beta:
        Rk = beta_k*spatial_covariance
        A = pilot_power*Rk + noise_var*identity
        filters.append(np.sqrt(pilot_power)*Rk @ np.linalg.solve(A, identity))
    return filters

def estimate_channel(H, beta, spatial_covariance, pilot_power, noise_var,
                     method, rng, precomputed_lmmse=None):
    m, k = H.shape
    pilot_noise = (rng.normal(size=(m, k)) + 1j*rng.normal(size=(m, k)))*np.sqrt(noise_var/2.0)
    Yp = np.sqrt(pilot_power)*H + pilot_noise
    method = method.upper()
    if method == "LS":
        return Yp/np.sqrt(pilot_power)
    if method == "LMMSE":
        filters = precomputed_lmmse
        if filters is None:
            filters = make_lmmse_filters(beta, spatial_covariance, pilot_power, noise_var)
        estimate = np.zeros_like(H)
        for user in range(k):
            estimate[:, user] = filters[user] @ Yp[:, user]
        return estimate
    raise ValueError("method must be LS or LMMSE")
