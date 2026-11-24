import numpy as np
from src.modem import qam_mod, qam_demod, bits_per_symbol
from src.channel import spatial_correlation, large_scale_fading, generate_taps, frequency_response
from src.estimation import estimate_channel
from src.detectors import make_combiner, post_detection_sinr

def test_qam_round_trip_all_orders():
    rng = np.random.default_rng(8)
    for q in (4,16,64):
        bits = rng.integers(0,2,400*bits_per_symbol(q),dtype=np.int8)
        assert np.array_equal(bits, qam_demod(qam_mod(bits,q),q))

def test_channel_and_tap_shapes():
    rng=np.random.default_rng(3)
    _,_,beta=large_scale_fading(8,rng)
    R,Rs=spatial_correlation(64,.35)
    taps=generate_taps(beta,Rs,np.ones(4)/4,rng)
    H=frequency_response(taps,0,8)
    assert taps.shape==(4,64,8) and H.shape==(64,8)
    assert np.allclose(R,R.conj().T)

def test_estimators_and_receivers():
    rng=np.random.default_rng(4)
    M,K=16,4
    H=(rng.normal(size=(M,K))+1j*rng.normal(size=(M,K)))/np.sqrt(2)
    beta=np.ones(K); R,_=spatial_correlation(M,.2)
    for method in ("LS","LMMSE"):
        Hhat=estimate_channel(H,beta,R,1.0,1.0,method,rng)
        assert Hhat.shape==H.shape
        for det in ("MRC","ZF","MMSE"):
            W=make_combiner(Hhat,.1,det)
            sinr=post_detection_sinr(W,H,.1)
            assert W.shape==(K,M) and sinr.shape==(K,)
            assert np.all(np.isfinite(sinr)) and np.all(sinr>=0)
