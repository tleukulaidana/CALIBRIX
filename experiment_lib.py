import numpy as np
from scipy.ndimage import gaussian_filter
# ---- corruptions (images in [0,1]) ----
def noise(X, s, rng): return np.clip(X + rng.normal(0, s, X.shape), 0, 1)
def blur(X, s, rng): return np.stack([gaussian_filter(x, s) for x in X])
def fog(X, a, rng):
    haze = np.stack([gaussian_filter(rng.random((8, 8)), 2.0) for _ in X])
    haze = (haze - haze.min((1, 2), keepdims=True)) / (np.ptp(haze, (1, 2), keepdims=True) + 1e-9)
    haze = 0.5 + 0.4 * haze
    return np.clip((1 - a) * X + a * haze, 0, 1)
CORR = {"noise": (noise, [0.15, 0.30, 0.45]), "blur": (blur, [0.6, 1.0, 1.5]), "fog": (fog, [0.3, 0.5, 0.7])}

