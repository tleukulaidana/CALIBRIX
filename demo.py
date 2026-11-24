"""1-minute demo: foggy digits -> ensemble -> accept or reject. Run: python demo.py"""
import time, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from cnn import Net, softmax
from experiment_lib import fog

t0 = time.time()
D = load_digits(); X = D.images / 16.0; y = D.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print("Training 5 small CNNs (a deep ensemble)...")
nets = [Net(np.random.default_rng(k)) for k in range(5)]
for k, n in enumerate(nets): n.fit(Xtr, ytr, np.random.default_rng(50 + k), epochs=40)
rng = np.random.default_rng(1)
Xf = fog(Xte, 0.7, rng)                                   # heavy synthetic fog
P = np.mean([softmax(n.logits(Xf)) for n in nets], 0)
conf, pred = P.max(1), P.argmax(1); ok = pred == yte
thr = np.quantile(conf, 0.20); keep = conf >= thr         # reject the 20% least confident
print(f"Clean accuracy        : {np.mean(np.mean([softmax(n.logits(Xte)) for n in nets],0).argmax(1)==yte):.1%}")
print(f"Foggy accuracy (all)  : {ok.mean():.1%}   (answers every image)")
print(f"Foggy accuracy (kept) : {ok[keep].mean():.1%}   (keeps 80%, rejects 20% least confident)")
print(f"Wrong answers caught  : {(~ok & ~keep).sum()} of {(~ok).sum()} errors were rejected")
idx = np.r_[np.where(~keep)[0][:6], np.where(keep)[0][:6]]   # first 6 rejected, first 6 accepted (no cherry-picking)
fig, ax = plt.subplots(2, 6, figsize=(11, 5.0))
for a, i in zip(ax.ravel(), idx):
    a.imshow(Xf[i], cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    col = "#c0392b" if not keep[i] else "#27ae60"
    for s in a.spines.values(): s.set_edgecolor(col); s.set_linewidth(4)
    a.set_title(f"{'REJECT' if not keep[i] else 'ACCEPT'}  conf {conf[i]:.2f}\npred {pred[i]} / true {yte[i]}", fontsize=9, color=col)
fig.suptitle("Heavy fog: the ensemble rejects low-confidence images", fontsize=12); fig.tight_layout(h_pad=3.5)
fig.savefig("demo_output.png", dpi=170); print(f"Saved demo_output.png  ({time.time()-t0:.0f}s total)")
