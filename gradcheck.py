import numpy as np
from cnn import Net, softmax
rng = np.random.default_rng(0)
net = Net(rng, p=0.5)
X = rng.random((6, 8, 8)); y = rng.integers(0, 10, 6)
# fixed dropout mask: reseed rng identically each call
def L(): 
    r = np.random.default_rng(5); return net.loss_grads(X, y, r, wd=0)
loss, g = L()
maxerr = 0
for k in ["Wc", "W2", "W3", "bc"]:
    P = net.P[k]
    for _ in range(6):
        idx = tuple(rng.integers(0, s) for s in P.shape)
        old = P[idx]; eps = 1e-5
        P[idx] = old + eps; lp = L()[0]
        P[idx] = old - eps; lm = L()[0]
        P[idx] = old
        num = (lp - lm) / (2 * eps); an = g[k][idx]
        maxerr = max(maxerr, abs(num - an) / (abs(num) + abs(an) + 1e-9))
print("max relative gradient error:", maxerr)
