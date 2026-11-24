import numpy as np

def bits_per_symbol(order):
    if order not in (4, 16, 64):
        raise ValueError("Supported square-QAM orders are 4, 16 and 64.")
    return int(np.log2(order))

def gray_to_binary(gray):
    binary = np.asarray(gray, dtype=int).copy()
    shift = 1
    while shift < 8:
        binary ^= binary >> shift
        shift *= 2
    return binary

def qam_mod(bits, order):
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    k = bits_per_symbol(order)
    n_axis_bits = k//2
    levels = int(np.sqrt(order))
    if len(bits) % k:
        raise ValueError("Bit count must be divisible by log2(Q).")
    grouped = bits.reshape(-1, k)
    weights = 2**np.arange(n_axis_bits-1, -1, -1)
    binary_i = (grouped[:, :n_axis_bits]*weights).sum(axis=1).astype(int)
    binary_q = (grouped[:, n_axis_bits:]*weights).sum(axis=1).astype(int)
    gray_i = binary_i ^ (binary_i >> 1)
    gray_q = binary_q ^ (binary_q >> 1)
    normalization = np.sqrt(2.0*(order-1)/3.0)
    return ((2*gray_i-(levels-1)) + 1j*(2*gray_q-(levels-1)))/normalization

def qam_demod(symbols, order):
    z = np.asarray(symbols).reshape(-1)
    k = bits_per_symbol(order)
    n_axis_bits = k//2
    levels = int(np.sqrt(order))
    weights = 2**np.arange(n_axis_bits-1, -1, -1)
    normalization = np.sqrt(2.0*(order-1)/3.0)
    def decode_axis(values):
        gray = np.rint((values*normalization+(levels-1))/2).astype(int)
        return gray_to_binary(np.clip(gray, 0, levels-1))
    binary_i = decode_axis(z.real)
    binary_q = decode_axis(z.imag)
    out = np.zeros((len(z), k), dtype=np.int8)
    for j in range(n_axis_bits):
        out[:, j] = (binary_i//weights[j]) % 2
        out[:, n_axis_bits+j] = (binary_q//weights[j]) % 2
    return out.reshape(-1)
