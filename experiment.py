"""CALIBRIX prototype: uncertainty calibration + reject option under synthetic corruptions.
Dataset: scikit-learn Digits (UCI optical recognition of handwritten digits, 8x8). Pure-NumPy CNN.
Pre-specified success criteria (fixed BEFORE running, see SUCCESS below)."""
import json, time, numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from cnn import Net, softmax

SEEDS = list(range(10)); N_ENS = 5; EPOCHS = 40; MC_T = 30
SUCCESS = {"ECE_rel_reduction_min": 0.30, "AURC_rel_reduction_min": 0.10}  # best method vs softmax, mean over 9 corrupted conditions

# ---- corruptions (images in [0,1]) ----
def noise(X, s, rng): return np.clip(X + rng.normal(0, s, X.shape), 0, 1)
def blur(X, s, rng): return np.stack([gaussian_filter(x, s) for x in X])
def fog(X, a, rng):
    haze = np.stack([gaussian_filter(rng.random((8, 8)), 2.0) for _ in X])
    haze = (haze - haze.min((1, 2), keepdims=True)) / (np.ptp(haze, (1, 2), keepdims=True) + 1e-9)
    haze = 0.5 + 0.4 * haze
    return np.clip((1 - a) * X + a * haze, 0, 1)
CORR = {"noise": (noise, [0.15, 0.30, 0.45]), "blur": (blur, [0.6, 1.0, 1.5]), "fog": (fog, [0.3, 0.5, 0.7])}

# ---- metrics ----
def ece(p, y, nb=10):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y)
    bins = np.linspace(0, 1, nb + 1); e = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e
def nll(p, y): return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
def risk_cov(p, y):
    conf, err = p.max(1), (p.argmax(1) != y).astype(float)
    o = np.argsort(-conf, kind="stable"); cum = np.cumsum(err[o]) / np.arange(1, len(y) + 1)
    return cum
def aurc(p, y): return risk_cov(p, y).mean()
def acc_at_cov(p, y, c=0.8):
    cum = risk_cov(p, y); k = int(round(c * len(y))); return 1 - cum[k - 1]
def metrics(p, y):
    return dict(acc=float((p.argmax(1) == y).mean()), ece=float(ece(p, y)), nll=float(nll(p, y)),
                aurc=float(aurc(p, y)), acc80=float(acc_at_cov(p, y)))

def fit_T(logits, y):
    f = lambda t: nll(softmax(logits / t), y)
    return minimize_scalar(f, bounds=(0.05, 10), method="bounded").x

D = load_digits(); Xall = D.images / 16.0; yall = D.target
METHODS = ["softmax", "TS-clean", "TS-aug", "MC-dropout", "ensemble"]
conds = [("clean", 0)] + [(c, s) for c in CORR for s in (1, 2, 3)]
res = {m: {f"{c}{s}" if c != "clean" else "clean": [] for c, s in conds} for m in METHODS}
extra = {"T_clean": [], "T_aug": [], "rc_curves": {}, "reliab": {}}
t0 = time.time()
for seed in SEEDS:
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xall, yall, test_size=0.40, random_state=seed, stratify=yall)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.625, random_state=seed, stratify=ytmp)
    rng = np.random.default_rng(1000 + seed)
    nets = []
    for k in range(N_ENS):
        n = Net(np.random.default_rng(seed * 100 + k)); n.fit(Xtr, ytr, np.random.default_rng(seed * 100 + k + 50), epochs=EPOCHS); nets.append(n)
    net0 = nets[0]
    # temperature scaling: clean val vs corrupted val (noise s2 + blur s2 only -> fog is UNSEEN)
    Tc = fit_T(net0.logits(Xva), yva)
    Xaug = np.concatenate([noise(Xva, CORR["noise"][1][1], rng), blur(Xva, CORR["blur"][1][1], rng)])
    yaug = np.concatenate([yva, yva]); Ta = fit_T(net0.logits(Xaug), yaug)
    extra["T_clean"].append(float(Tc)); extra["T_aug"].append(float(Ta))
    for c, s in conds:
        Xc = Xte if c == "clean" else CORR[c][0](Xte, CORR[c][1][s - 1], rng)
        key = "clean" if c == "clean" else f"{c}{s}"
        lg = net0.logits(Xc)
        P = {"softmax": softmax(lg), "TS-clean": softmax(lg / Tc), "TS-aug": softmax(lg / Ta),
             "MC-dropout": net0.mc_probs(Xc, rng, MC_T),
             "ensemble": np.mean([softmax(n.logits(Xc)) for n in nets], 0)}
        for m in METHODS: res[m][key].append(metrics(P[m], yte))
        if key in ("noise3", "fog3"):
            extra["rc_curves"].setdefault(key, {m: [] for m in METHODS})
            for m in METHODS: extra["rc_curves"][key][m].append(risk_cov(P[m], yte).tolist())
        if key in ("noise2", "fog2", "fog3", "clean"):
            extra["reliab"].setdefault(key, {m: {"conf": [], "corr": []} for m in METHODS})
            for m in METHODS:
                extra["reliab"][key][m]["conf"] += P[m].max(1).tolist(); extra["reliab"][key][m]["corr"] += (P[m].argmax(1) == yte).tolist()
    print(f"seed {seed} done  {time.time()-t0:.0f}s", flush=True)

# ---- aggregate ----
agg = {m: {k: {mm: [float(np.mean([r[mm] for r in v])), float(np.std([r[mm] for r in v]))] for mm in v[0]} for k, v in res[m].items()} for m in METHODS}
corr_keys = [k for k in agg["softmax"] if k != "clean"]
summ = {m: {mm: [float(np.mean([agg[m][k][mm][0] for k in corr_keys]))] for mm in ["acc", "ece", "nll", "aurc", "acc80"]} for m in METHODS}
# per-seed paired means over corrupted conditions (for sd across seeds)
for m in METHODS:
    for mm in ["acc", "ece", "nll", "aurc", "acc80"]:
        per_seed = [np.mean([res[m][k][i][mm] for k in corr_keys]) for i in range(len(SEEDS))]
        summ[m][mm] = [float(np.mean(per_seed)), float(np.std(per_seed))]
# per-corruption-type summary
bytype = {m: {c: {mm: float(np.mean([agg[m][f"{c}{s}"][mm][0] for s in (1, 2, 3)])) for mm in ["acc","ece","aurc","acc80"]} for c in CORR} for m in METHODS}
best_ece = min(METHODS, key=lambda m: summ[m]["ece"][0]); best_aurc = min(METHODS, key=lambda m: summ[m]["aurc"][0])
ece_red = 1 - summ[best_ece]["ece"][0] / summ["softmax"]["ece"][0]; aurc_red = 1 - summ[best_aurc]["aurc"][0] / summ["softmax"]["aurc"][0]
verdict = {"best_ece_method": best_ece, "ECE_rel_reduction": float(ece_red), "ECE_target_met": bool(ece_red >= SUCCESS["ECE_rel_reduction_min"]),
           "best_aurc_method": best_aurc, "AURC_rel_reduction": float(aurc_red), "AURC_target_met": bool(aurc_red >= SUCCESS["AURC_rel_reduction_min"])}
# time cost (relative inference cost): softmax=1, MC=30, ens=5
json.dump(dict(res=res, agg=agg, summ=summ, bytype=bytype, verdict=verdict, extra=extra, success=SUCCESS, seeds=SEEDS, n_ens=N_ENS, mc_T=MC_T, epochs=EPOCHS,
               split=dict(train=len(Xtr), val=len(Xva), test=len(Xte))), open("results.json", "w"))
print(json.dumps(summ, indent=1)); print(json.dumps(verdict, indent=1))
print("T_clean mean", np.mean(extra["T_clean"]), "T_aug mean", np.mean(extra["T_aug"]))
print("clean:", {m: agg[m]["clean"] for m in METHODS})
