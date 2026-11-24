"""Tiny CNN in pure NumPy (conv3x3 -> ReLU -> dense -> dropout -> dense) with Adam."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

class Net:
    def __init__(self, rng, nf=8, nh=64, p=0.5):
        self.nf, self.nh, self.p = nf, nh, p
        he = lambda fi, shape: rng.normal(0, np.sqrt(2.0 / fi), shape)
        self.P = {
            "Wc": he(9, (9, nf)), "bc": np.zeros(nf),
            "W2": he(64 * nf, (64 * nf, nh)), "b2": np.zeros(nh),
            "W3": he(nh, (nh, 10)), "b3": np.zeros(10),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def _cols(self, X):
        Xp = np.pad(X, ((0, 0), (1, 1), (1, 1)))
        return swv(Xp, (3, 3), axis=(1, 2)).reshape(len(X) * 64, 9)

    def forward(self, X, drop=False, rng=None):
        P = self.P
        cols = self._cols(X)
        a1 = cols @ P["Wc"] + P["bc"]
        r1 = np.maximum(a1, 0)
        flat = r1.reshape(len(X), -1)
        a2 = flat @ P["W2"] + P["b2"]
        h = np.maximum(a2, 0)
        mask = None
        if drop:
            mask = (rng.random(h.shape) > self.p) / (1 - self.p)
            h = h * mask
        logits = h @ P["W3"] + P["b3"]
        return logits, (cols, a1, flat, a2, h, mask)

    def loss_grads(self, X, y, rng, wd=1e-4):
        P = self.P
        logits, (cols, a1, flat, a2, h, mask) = self.forward(X, True, rng)
        pr = softmax(logits)
        n = len(X)
        loss = -np.log(pr[np.arange(n), y] + 1e-12).mean()
        d = pr.copy(); d[np.arange(n), y] -= 1; d /= n
        g = {}
        g["W3"] = h.T @ d + wd * P["W3"]; g["b3"] = d.sum(0)
        dh = d @ P["W3"].T
        dh = dh * mask * (a2 > 0)
        g["W2"] = flat.T @ dh + wd * P["W2"]; g["b2"] = dh.sum(0)
        dflat = (dh @ P["W2"].T).reshape(n * 64, self.nf) * (a1 > 0)
        g["Wc"] = cols.T @ dflat + wd * P["Wc"]; g["bc"] = dflat.sum(0)
        return loss, g

    def adam(self, g, lr=2e-3, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.P:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, X, y, rng, epochs=40, bs=32):
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), bs):
                b = idx[i:i + bs]
                _, g = self.loss_grads(X[b], y[b], rng)
                self.adam(g)

    def logits(self, X):
        return self.forward(X)[0]

    def mc_probs(self, X, rng, T=30):
        return np.mean([softmax(self.forward(X, True, rng)[0]) for _ in range(T)], 0)
