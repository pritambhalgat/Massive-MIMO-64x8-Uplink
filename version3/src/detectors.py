import numpy as np

def make_combiner(H_est, noise_eff, detector):
    _, k = H_est.shape
    gram = H_est.conj().T @ H_est
    detector = detector.upper()
    if detector == "MRC":
        return H_est.conj().T
    if detector == "ZF":
        loading = 1e-10*max(1.0, float(np.real(np.trace(gram))/k))
        return np.linalg.solve(gram + loading*np.eye(k), H_est.conj().T)
    if detector == "MMSE":
        loading = 1e-10*max(1.0, float(np.real(np.trace(gram))/k))
        return np.linalg.solve(gram + (noise_eff+loading)*np.eye(k), H_est.conj().T)
    raise ValueError("detector must be MRC, ZF or MMSE")

def post_detection_sinr(W, H_true, noise_eff):
    effective = W @ H_true
    desired = np.abs(np.diag(effective))**2
    interference = np.maximum(np.sum(np.abs(effective)**2, axis=1)-desired, 0.0)
    noise = noise_eff*np.sum(np.abs(W)**2, axis=1)
    return desired/np.maximum(interference+noise, 1e-15)
