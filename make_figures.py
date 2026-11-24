import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
r = json.load(open("results.json")); M = ["softmax", "TS-clean", "TS-aug", "MC-dropout", "ensemble"]
COL = {"softmax": "#8c8c8c", "TS-clean": "#9ecae1", "TS-aug": "#3182bd", "MC-dropout": "#e8a064", "ensemble": "#2c2f54"}
LAB = {"softmax": "Softmax (baseline)", "TS-clean": "Temp. scaling (clean)", "TS-aug": "Temp. scaling (corrupted val.)", "MC-dropout": "MC-dropout", "ensemble": "Ensemble (5)"}

# Fig 1: ECE / AURC by corruption
keys = [k for k in r["agg"]["softmax"] if k != "clean"]
groups = ["noise", "blur", "fog", "mean"]
def val(m, g, mm):
    return r["summ"][m][mm][0] if g == "mean" else r["bytype"][m][g][mm]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for a, mm, t in zip(ax, ["ece", "aurc"], ["ECE (lower is better)", "AURC (lower is better)"]):
    w = 0.16
    for i, m in enumerate(M):
        a.bar(np.arange(4) + (i - 2) * w, [val(m, g, mm) for g in groups], w, color=COL[m], label=LAB[m])
    a.set_xticks(range(4)); a.set_xticklabels(["Noise", "Blur", "Fog", "Mean"]); a.set_title(t, fontsize=9.5, loc="left")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=5, handlelength=1.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig("fig1_bars.png", dpi=220); plt.close()

# Fig 2: risk-coverage + reliability (fog, severity 2)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
rc = r["extra"]["rc_curves"]["fog3"]; n = len(rc["softmax"][0]); cov = np.arange(1, n + 1) / n
for m in ["softmax", "MC-dropout", "ensemble"]:
    ax[0].plot(cov, np.mean(rc[m], 0), color=COL[m], lw=1.6, label=LAB[m])
ax[0].set_xlabel("Coverage (fraction of images answered)"); ax[0].set_ylabel("Risk (error on answered)"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("Risk-coverage, fog severity 3 (mean of 10 splits)", fontsize=9.5, loc="left")
rl = r["extra"]["reliab"]["fog3"]; bins = np.linspace(0, 1, 11); ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
for m in ["softmax", "ensemble"]:
    c = np.array(rl[m]["conf"]); k = np.array(rl[m]["corr"], float); xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (c > lo) & (c <= hi)
        if mk.sum() >= 15: xs.append(c[mk].mean()); ys.append(k[mk].mean())
    ax[1].plot(xs, ys, "o-", color=COL[m], ms=4, lw=1.4, label=LAB[m])
ax[1].set_xlabel("Confidence"); ax[1].set_ylabel("Accuracy"); ax[1].legend(fontsize=7, frameon=False, loc="upper left")
ax[1].set_title("Reliability, fog severity 3 (10 splits pooled)", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig("fig2_rc_rel.png", dpi=220); plt.close()

# Fig 3: Gantt (18 months)
WP = [("WP1 Literature review and protocol", 1, 4, "D1 (M4)"), ("WP2 Benchmark and corruption generator", 2, 8, "D2 (M8)"),
      ("WP3 Calibration and reject methods", 5, 14, "D3 (M14)"), ("WP4 Validation and cost study", 11, 17, "D4 (M17)"),
      ("WP5 Dissemination, transfer, management", 1, 18, "D5 (M18)")]
fig, ax = plt.subplots(figsize=(7.4, 2.15))
for i, (n, s, e, d) in enumerate(WP):
    ax.barh(i, e - s + 1, left=s - 0.5, color="#e8a064", edgecolor="white", height=0.62)
    ax.text(18.9, i, d, va="center", fontsize=7.5)
for mm, lab in [(4, "M1"), (8, "M2"), (14, "M3"), (17, "M4")]:
    ax.plot(mm + 0.5, -0.62, marker="D", color="#2c2f54", ms=5, clip_on=False)
ax.set_yticks(range(5)); ax.set_yticklabels([w[0] for w in WP], fontsize=7.8); ax.invert_yaxis()
ax.set_xlim(0.5, 18.5); ax.set_xticks(range(1, 19)); ax.set_xlabel("Project month  (diamonds: milestones M1 = M4, M2 = M8, M3 = M14, M4 = M17)", fontsize=7.8)
ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig("fig3_gantt.png", dpi=220); plt.close()

# Fig 4: pipeline (for slides)
fig, ax = plt.subplots(figsize=(10, 2.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 26)
boxes = [(0.5, "Input image\n(clean, noisy,\nblurry, foggy)"), (21, "CNN\n(single or\nensemble of 5)"), (41.5, "Confidence =\nmax class\nprobability"), (62, "Threshold:\nreject least\nconfident 20%")]
for x, t in boxes:
    ax.add_patch(FancyBboxPatch((x, 5), 18, 16, boxstyle="round,pad=0.3", fc="#eef0f7", ec="#2c2f54", lw=1.6)); ax.text(x + 8.75, 13, t, ha="center", va="center", fontsize=12.5)
for x in (18.6, 39.1, 59.6): ax.add_patch(FancyArrowPatch((x, 13), (x + 2.4, 13), arrowstyle="-|>", mutation_scale=16, color="#2c2f54"))
ax.add_patch(FancyBboxPatch((83, 13.5), 16, 10, boxstyle="round,pad=0.3", fc="#d5f0dd", ec="#27ae60", lw=1.6)); ax.text(91, 18.5, "ACCEPT:\nanswer", ha="center", va="center", fontsize=11.5)
ax.add_patch(FancyBboxPatch((83, 1.5), 16, 10, boxstyle="round,pad=0.3", fc="#f8d7d3", ec="#c0392b", lw=1.6)); ax.text(91, 6.5, "REJECT:\nask human", ha="center", va="center", fontsize=11.5)
ax.add_patch(FancyArrowPatch((80.2, 15), (82.6, 18.5), arrowstyle="-|>", mutation_scale=16, color="#27ae60")); ax.add_patch(FancyArrowPatch((80.2, 11), (82.6, 7), arrowstyle="-|>", mutation_scale=16, color="#c0392b"))
fig.savefig("fig4_pipeline.png", dpi=220, bbox_inches="tight"); plt.close()
print("ok")
