"""Builds A12 PDF = proposal (A4) + slides (16:9). All numbers are read from results.json."""
import json, io, numpy as np
from scipy.stats import wilcoxon
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, Flowable)
from reportlab.pdfgen import canvas as rl_canvas
from pypdf import PdfReader, PdfWriter


AUTHOR = "[Aidana Tileukul]"
VIDEO_URL = ""          
CODE_URL = ""           
OUT = "A12_Surname_Name.pdf"

NAVY, ORANGE, LIGHT = colors.HexColor("#2c2f54"), colors.HexColor("#e8a064"), colors.HexColor("#eef0f7")
r = json.load(open("results.json")); res = r["res"]; S = r["summ"]; V = r["verdict"]; A = r["agg"]
M = ["softmax", "TS-clean", "TS-aug", "MC-dropout", "ensemble"]
NAME = {"softmax": "Softmax (baseline)", "TS-clean": "Temp. scaling, clean val.", "TS-aug": "Temp. scaling, corrupted val.", "MC-dropout": "MC-dropout (30 passes)", "ensemble": "Deep ensemble (5 CNNs)"}
keys = [k for k in res["softmax"] if k != "clean"]
def per_seed(m, mm): return np.array([np.mean([res[m][k][i][mm] for k in keys]) for i in range(10)])
def paired(m, mm):
    d = per_seed("softmax", mm) - per_seed(m, mm); return int((d > 0).sum()), float(wilcoxon(d).pvalue)
rel = lambda m, mm: 100 * (S[m][mm][0] / S["softmax"][mm][0] - 1)
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
e_w, e_p = paired("ensemble", "ece"); a_w, a_p = paired("ensemble", "aurc"); mc_w, mc_p = paired("MC-dropout", "aurc"); ta_w, ta_p = paired("TS-aug", "aurc")
Tc, Ta = np.mean(r["extra"]["T_clean"]), np.mean(r["extra"]["T_aug"])
bt = r["bytype"]; ag = lambda m, k, mm: A[m][k][mm][0]
VID = VIDEO_URL or "[PASTE VIDEO LINK HERE]"; CODE = CODE_URL or "[PASTE CODE LINK HERE]"

# ------------------------------------------------------------------ styles ----
B = ParagraphStyle("B", fontName="Helvetica", fontSize=9.3, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=3)
H = ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=10.4, leading=12.5, textColor=NAVY, spaceBefore=6, spaceAfter=2.5)
H2 = ParagraphStyle("H2", parent=H, fontSize=9.2, spaceBefore=3, spaceAfter=1.5)
T = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, spaceAfter=2)
SM = ParagraphStyle("SM", parent=B, fontSize=8.1, leading=9.9, alignment=TA_LEFT, spaceAfter=0)
SMB = ParagraphStyle("SMB", parent=SM, fontName="Helvetica-Bold")
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=9.6, textColor=colors.HexColor("#444444"), alignment=TA_LEFT, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
P = lambda t, s=B: Paragraph(t, s)
def tbl(rows, widths, head=True, zebra=True, fs=SM):
    data = [[Paragraph(str(c), SMB if (head and i == 0) else fs) if not isinstance(c, Flowable) else c for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
    t.setStyle(TableStyle(st)); return t
def img(path, w):
    from PIL import Image as PI
    iw, ih = PI.open(path).size; return Image(path, width=w, height=w * ih / iw)
class Marker(Flowable):
    pages = {}
    def __init__(self, name): super().__init__(); self.name = name
    def wrap(self, *a): return (0, 0)
    def draw(self): Marker.pages[self.name] = self.canv.getPageNumber()

# ------------------------------------------------------------------ proposal ----
W = A4[0] - 3.4 * cm
def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.2); c.setFillColor(colors.HexColor("#666666"))
    c.drawString(1.7 * cm, 1.0 * cm, "Research Tools and Methods - Assignment 12 - CALIBRIX - " + AUTHOR); c.drawRightString(A4[0] - 1.7 * cm, 1.0 * cm, f"Page {d.page}")
    c.setStrokeColor(ORANGE); c.setLineWidth(1.2); c.line(1.7 * cm, A4[1] - 1.1 * cm, A4[0] - 1.7 * cm, A4[1] - 1.1 * cm); c.restoreState()
doc = BaseDocTemplate("proposal.pdf", pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="CALIBRIX - Assignment 12", author=AUTHOR)
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, W, A4[1] - 3.0 * cm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
st = []
vlink = f'<link href="{VIDEO_URL}" color="#1a5fb4">{VIDEO_URL}</link>' if VIDEO_URL else f'<font backColor="#fff3b0"><b>{VID}</b></font>'
clink = f'<link href="{CODE_URL}" color="#1a5fb4">{CODE_URL}</link>' if CODE_URL else f'<font backColor="#fff3b0"><b>{CODE}</b></font>'
# 1 title
st += [P("CALIBRIX: Calibrated Uncertainty and a Reject Option for Image Classifiers under Visual Corruptions", T),
       P(f"<b>Acronym:</b> CALIBRIX &nbsp;|&nbsp; <b>Applicant:</b> {AUTHOR} &nbsp;|&nbsp; <b>Course:</b> Research Tools and Methods, SDU University &nbsp;|&nbsp; <b>Duration:</b> 18 months &nbsp;|&nbsp; <b>Type:</b> applied research &nbsp;|&nbsp; <b>TRL:</b> 3 now, 4 target", SM),
       Spacer(1, 2), P(f"<b>Video presentation (3-4 min):</b> {vlink} &nbsp;&nbsp; <b>Code and data:</b> {clink}", SM), Spacer(1, 3)]
st += [P("1. Title, acronym and abstract", H)]
abstract = (f"Image classifiers used outdoors meet fog, noise and blur, and they often stay confident when they are wrong. This is risky when decisions are automatic, for example in traffic cameras or industrial inspection. "
 f"CALIBRIX studies cheap ways to let a classifier know when it does not know: softmax confidence, temperature scaling, MC-dropout and deep ensembles, combined with a reject option that skips the least confident inputs. "
 f"We measure calibration (ECE) and the risk-coverage trade-off (AURC) under synthetic corruptions. A TRL-3 prototype (NumPy CNN, 8x8 handwritten digits, 3 corruption types x 3 severities, 10 random splits) shows that a 5-model ensemble lowered mean ECE by {abs(rel('ensemble','ece')):.0f}% ({f3(S['softmax']['ece'][0])} to {f3(S['ensemble']['ece'][0])}) and mean AURC by {abs(rel('ensemble','aurc')):.0f}% ({f4(S['softmax']['aurc'][0])} to {f4(S['ensemble']['aurc'][0])}) compared with softmax. "
 f"Temperature scaling fitted on clean data did not help under corruption, and MC-dropout did not improve selective prediction. The pre-set ECE target (-30%) was missed and the AURC gain was not statistically significant (p = {a_p:.2f}). "
 f"The 18-month project moves the study to CIFAR-10-C-style benchmarks and CNNs such as ResNet (TRL 4), adds a small real-fog test set, reports compute cost and releases an open-source toolkit.")
assert len(abstract.split()) <= 200, len(abstract.split())
st += [P(abstract + f" <i>({len(abstract.split())} words)</i>")]

# 2 relevance
st += [P("2. Relevance, problem statement and type of R&amp;D", H),
 P("Deep networks that classify images are accurate on clean test data, but accuracy drops when images are degraded by fog, sensor noise or blur, and the model's confidence often does not drop with it [1], [6]. "
   "A wrong answer given with high confidence is worse than 'I don't know', because nobody checks it. This matters for outdoor cameras (traffic, safety, agriculture) and for industrial inspection in a country with harsh winter weather and dust, where fog, snow and low light are normal. "
   "<b>Problem:</b> how well do cheap uncertainty methods stay calibrated and support a reliable reject option when image quality degrades, and which one is worth its compute cost?"),
 P("<b>Type of R&amp;D: applied research.</b> The work answers a specific practical question and produces methods, evidence and a proof of concept, not new theory (basic research) and not a deployed system (experimental development). Justification against the Frascati criteria [11]:"),
 tbl([["Criterion", "How CALIBRIX meets it"],
  ["Novel", "New evidence on reject-option behaviour per corruption type, and on whether a calibration step fitted on a few synthetic corruptions transfers to an unseen one (TS-aug, tested on fog)."],
  ["Creative", "Uses cheap synthetic corruptions as a proxy for calibration data, instead of only clean validation data."],
  ["Uncertain", "The outcome can fail. In the prototype, temperature scaling on clean data did not help, MC-dropout hurt AURC, and the ECE target was missed."],
  ["Systematic", "Success criteria fixed before running; fixed seeds and splits; 5 work packages with milestones; versioned code."],
  ["Transferable / reproducible", "Open code; one command reproduces all numbers in about 80 s on one CPU core; results stored as JSON."]],
  [3.3 * cm, W - 3.3 * cm])]

# 3 state of art
st += [P("3. State of the art, gap, and TRL", H),
 P("Modern networks tend to be overconfident, and temperature scaling is a simple and effective post-hoc fix on in-distribution data [1]. Softmax confidence is the standard baseline for detecting errors [7] and for selective classification, where the model may reject inputs to lower its risk [5]. "
   "Two popular uncertainty methods are MC-dropout [3] and deep ensembles [4]. Benchmarks of common corruptions such as noise, blur and fog were introduced in [2]. Ovadia et al. [6] reported that calibration degrades as the data shift grows and that ensembles were among the most robust methods they tested, while calibration tuned on in-distribution data was less reliable under shift."),
 P("<b>Gap (our reading, to be confirmed by the systematic review in WP1).</b> Results are usually reported in aggregate. Less is known about (i) per-corruption results at a fixed coverage, which is what a user with a reject option cares about; (ii) whether a calibration step fitted on a few synthetic corruptions transfers to unseen ones; and (iii) which method is worth its compute cost for low-resource users. "
   "<b>TRL:</b> current TRL 3 (experimental proof of concept, Section 5); target TRL 4 (validated in the laboratory on CIFAR-10-C-style data [2], [8] and a small real-fog set) at month 18 [12].")]

# 4 aim, objectives
st += [P("4. Aim and SMART objectives", H),
 P("<b>Aim:</b> make image classifiers safer under visual degradation by finding the cheapest method that gives reliable confidence and a useful reject option."),
 tbl([["#", "SMART objective", "Measure"],
  ["O1", "By month 8, build an open benchmark with 3 corruption families x 5 severities on CIFAR-10 and a real-fog test subset of at least 500 images, with fixed splits and seeds.", "Benchmark released; documented"],
  ["O2", "By month 14, compare at least 5 methods (softmax, 2 temperature-scaling variants, MC-dropout, deep ensemble) on 3 CNN architectures; target: at least 30% lower mean ECE and at least 10% lower mean AURC than softmax.", "ECE, AURC, paired tests over at least 10 seeds"],
  ["O3", "By month 17, show selective accuracy of at least 90% at 80% coverage on corruption severities 1-3 with the best method, and report training and inference cost of every method.", "Accuracy at 80% coverage; seconds and memory per image"],
  ["O4", "By month 18, release an open-source toolkit (MIT licence) that reproduces all results with one command, and submit at least one paper.", "Repository with DOI; submission"]],
  [0.8 * cm, W - 5.8 * cm, 5.0 * cm])]

# 5 methodology + prototype
st += [P("5. Methodology and prototype study (TRL 3)", H),
 P("<b>Design of the full project.</b> Train CNNs on CIFAR-10 [8]; evaluate on clean data and on noise, blur and fog at several severities [2]; compare the five methods; for each method reject the least confident inputs and measure the trade-off. Success is measured with: <b>ECE</b> (10 bins; gap between confidence and accuracy), <b>AURC</b> (area under the risk-coverage curve; lower means errors are concentrated in the rejected inputs), <b>accuracy at 80% coverage</b>, and NLL. Every comparison uses several random splits and paired tests."),
 P("<b>Prototype (what was actually run).</b> Compute and data access were limited, so the TRL-3 prototype tests the key uncertainty on a small scale: the scikit-learn Digits data [9], [10] (1,797 real handwritten digit images, 8x8 pixels; per split 1,078 train, 269 validation, 450 test), a small CNN written in NumPy (3x3 conv with 8 filters, dense 64, dropout 0.5; gradients checked numerically), and corruptions generated by our own code: Gaussian noise (sigma 0.15, 0.30, 0.45), Gaussian blur (sigma 0.6, 1.0, 1.5) and fog (haze weight 0.3, 0.5, 0.7). "
   f"Ten random stratified splits were run. Temperature scaling was fitted on clean validation images (TS-clean; T = {Tc:.2f}) or on noise+blur validation images (TS-aug; T = {Ta:.2f}), so that fog is <i>unseen</i> by TS-aug. The ensemble has 5 CNNs; MC-dropout uses 30 stochastic passes. "
   "<b>Success criteria, fixed in the code before the first run:</b> the best method lowers mean ECE by at least 30% and mean AURC by at least 10% versus softmax, averaged over the 9 corrupted conditions."),
 P("<b>Results.</b> Table 1 shows the mean over the 9 corrupted conditions (mean and standard deviation across the 10 splits). On clean data all methods reach about 98% accuracy and ECE of about 0.01 (MC-dropout: 0.049).", B)]
rows = [["Method", "Accuracy", "ECE", "AURC", "Acc. at 80% cov.", "ECE vs softmax", "AURC vs softmax"]]
for m in M:
    rows.append([NAME[m], f"{S[m]['acc'][0]:.3f} ± {S[m]['acc'][1]:.3f}", f"{f3(S[m]['ece'][0])} ± {f3(S[m]['ece'][1])}", f"{f4(S[m]['aurc'][0])} ± {f4(S[m]['aurc'][1])}",
                 f"{S[m]['acc80'][0]:.3f}", "-" if m == "softmax" else f"{rel(m,'ece'):+.0f}%", "-" if m == "softmax" else f"{rel(m,'aurc'):+.1f}%"])
st += [tbl(rows, [4.1 * cm, 2.35 * cm, 2.35 * cm, 2.6 * cm, 2.0 * cm, 1.9 * cm, W - 15.3 * cm]),
 P("Table 1. Mean over 9 corrupted conditions (3 corruption types x 3 severities), 10 random splits; lower ECE and AURC are better.", CAP),
 img("fig1_bars.png", W * 0.86), P("Figure 1. ECE and AURC by corruption type (mean over 3 severities and 10 splits).", CAP)]
st += [P("<b>Findings.</b>", B),
 P(f"- <b>Ensemble is best on both metrics.</b> ECE {rel('ensemble','ece'):+.0f}% (better in {e_w}/10 splits, Wilcoxon p = {e_p:.3f}) and AURC {rel('ensemble','aurc'):+.1f}% (better in {a_w}/10 splits, p = {a_p:.2f}). The AURC target (10%) is met on average but the effect is not statistically reliable; the ECE target (30%) is <b>not met</b>.", BL),
 P(f"- <b>Temperature scaling on clean data does not help under corruption</b> (ECE {f3(S['TS-clean']['ece'][0])} vs {f3(S['softmax']['ece'][0])}); it learns T = {Tc:.2f}, close to 1, because the model is already well calibrated on clean data. Fitting on corrupted validation data gives a small gain (ECE {rel('TS-aug','ece'):+.0f}%; AURC {rel('TS-aug','aurc'):+.1f}%, better in {ta_w}/10 splits, so tiny but consistent). On unseen fog, ECE was {f3(bt['TS-aug']['fog']['ece'])} vs {f3(bt['softmax']['fog']['ece'])}.", BL),
 P(f"- <b>MC-dropout is inconsistent.</b> It has the lowest ECE on noise ({f3(bt['MC-dropout']['noise']['ece'])} vs {f3(bt['softmax']['noise']['ece'])}) but the worst on blur and fog, worse ECE on clean data, and worse AURC (better in only {mc_w}/10 splits, p = {mc_p:.2f}), at 30x inference cost.", BL),
 P(f"- <b>Heavy fog is hard for every method.</b> At severity 3 accuracy is about {ag('softmax','fog3','acc'):.2f}; all methods are overconfident, but the ensemble halves the ECE ({f3(ag('ensemble','fog3','ece'))} vs {f3(ag('softmax','fog3','ece'))}).", BL),
 img("fig2_rc_rel.png", W * 0.86), P("Figure 2. Left: risk-coverage curves; right: reliability diagram (points below the dashed line mean overconfidence).", CAP),
 P("<b>TRL evidence and limitations.</b> TRL 3 is supported because the key uncertainty (do cheap methods keep confidence reliable under corruption?) was tested experimentally with a baseline, pre-set criteria and repeated runs. It is not TRL 4 because the data are small and low-resolution, the corruptions are synthetic, the CNN is tiny, the 450 test images per condition are few, and the 10 splits come from the same 1,797 images, so the standard deviations understate the true uncertainty. "
   "<b>Next steps:</b> CIFAR-10-C-style data and standard architectures (WP2-3), real fog images, cost measurements (WP4).")]

# 6 work plan
st += [P("6. Work plan (18 months)", H), img("fig3_gantt.png", W * 0.9),
 P("Figure 3. Gantt chart. Milestones: M1 protocol frozen (month 4); M2 benchmark ready (month 8); M3 go/no-go: ECE and AURC targets checked (month 14); M4 final validation (month 17).", CAP),
 tbl([["WP", "Main tasks", "Lead", "Deliverable"],
  ["WP1 (M1-4)", "Systematic literature review; evaluation protocol; success criteria", "PI", "D1 review and protocol (M4)"],
  ["WP2 (M2-8)", "Corruption generator; CIFAR-10-C-style benchmark; collect and check licences for 500+ real fog images", "ML engineer", "D2 open benchmark (M8)"],
  ["WP3 (M5-14)", "Implement and tune the 5 methods on 3 architectures; unseen-corruption transfer tests", "ML engineer", "D3 calibration and reject toolkit (M14)"],
  ["WP4 (M11-17)", "Validation on real fog; statistical tests; cost study (time, memory); decision rule for choosing a method", "PI + student", "D4 validation report (M17)"],
  ["WP5 (M1-18)", "Management, risks, ethics and data plan; papers; open-source release; technology transfer", "PI", "D5 papers and final report (M18)"]],
  [2.2 * cm, W - 9.4 * cm, 2.2 * cm, 5.0 * cm])]

# 7 impact
st += [P("7. Expected results and impact", H),
 P("<b>Results:</b> an open benchmark (D2), a toolkit with five calibration and reject methods (D3), a validation and cost report with a simple method-selection rule (D4), and at least one conference or workshop paper. "
   "<b>Scientific impact:</b> reproducible per-corruption evidence, including negative results (for example, when temperature scaling on clean data does not transfer). "
   "<b>Economic and societal impact:</b> outdoor vision systems that flag uncertain images for human review instead of failing silently, which lowers the cost of errors in traffic monitoring, safety and inspection. "
   "<b>Dissemination:</b> open repository with DOI, paper, seminar at SDU. <b>Technology-transfer route:</b> disclose results to the university technology transfer office before publication; release the toolkit under an open licence; run a pilot with a partner company (traffic or inspection cameras) once TRL 4 is reached; apply to a commercialisation grant to move toward TRL 5-6 [12].")]

# 8 team, budget
rowsB = [("Personnel: PI, 0.2 FTE x 18 months x 450", 1620, "Supervision, WP1/WP4/WP5 lead"), ("Personnel: ML engineer, 1.0 FTE x 18 x 350", 6300, "All implementation in WP2-3"),
         ("Personnel: student researcher, 0.5 FTE x 18 x 160", 1440, "Experiments, documentation, real-image collection"),
         ("Equipment: GPU workstation", 2800, "About 100+ training runs; shared university GPUs are limited"), ("Consumables: cloud compute, storage", 900, "Peaks and backups"),
         ("Travel and dissemination", 1600, "One international conference, open-access fee"), ("Subcontracting: real-image labelling and licence check", 600, "500+ images; cannot be done in-house in time")]
direct = sum(x[1] for x in rowsB); ovh = round(0.10 * direct); total = direct + ovh
rowsT = [["Cost category (thousand KZT)", "Amount", "Justification"]] + [[a, f"{b:,}", c] for a, b, c in rowsB] + [["Overheads (10%, fixed rate set by funder)", f"{ovh:,}", "University administration, utilities"], ["<b>Total</b>", f"<b>{total:,}</b>", "About %.1f million KZT" % (total / 1000)]]
st += [P("8. Planned team and budget", H),
 P("<b>Hypothetical team if funded:</b> principal investigator (applied machine learning, 0.2 FTE); ML engineer (1.0 FTE); student researcher (0.5 FTE, bachelor or master level). Roles are planned, not yet filled. Budget figures are indicative estimates."),
 KeepTogether([tbl(rowsT, [6.6 * cm, 2.2 * cm, W - 8.8 * cm]), P("Table 2. Budget by cost category (thousand KZT).", CAP)])]

# 9 risks, ethics
st += [P("9. Risk register, ethics, data management (FAIR) and IP", H),
 tbl([["Risk", "Likelihood", "Impact", "Mitigation"],
  ["Gains seen on digits do not transfer to CIFAR-10-C", "Medium", "High", "Go/no-go milestone M3; report negative results; keep ensemble as main candidate"],
  ["GPU workstation delivered late", "Medium", "High", "Order in month 1; cloud compute as backup"],
  ["Real fog images are scarce or have unclear licences", "Medium", "Medium", "Start the licence check in WP2; use only images with permissive licences; fall back to the synthetic benchmark"],
  ["Key team member leaves", "Low", "High", "Document code and methods; name a backup for each WP"],
  ["Methods over-fit to the chosen corruption types", "Medium", "Medium", "Hold out corruption types; report unseen-type results"],
  ["Ensemble too costly for low-resource users", "Medium", "Medium", "Report cost; test distillation of the ensemble as plan B"],
  ["Costs rise above budget", "Low", "Medium", "Review budget quarterly; prefer cloud over purchases if prices rise"]],
  [5.3 * cm, 1.7 * cm, 1.5 * cm, W - 8.5 * cm]),
 Spacer(1, 4), P("<b>Ethics.</b> No human participants or personal data. Datasets are public benchmarks. Real fog images must not show identifiable people or number plates; such images are removed or blurred. "
   "<b>Data management (FAIR) [13].</b> A data management plan lists all datasets and licences. Code and the generated benchmark are published in a public repository with a DOI (Zenodo), a README, fixed seeds and metadata, so that they are Findable, Accessible, Interoperable (open formats) and Reusable (open licence). "
   "<b>IP.</b> Ownership is agreed with the university before the start. Code is released under the MIT licence; no patent is planned because the results are methods and evidence, but the technology transfer office is consulted before publication.", B)]

# 10 AI statement
st += [P("10. AI-use statement", H),
 P("Generative AI (Claude, Anthropic) was used for brainstorming the topic, drafting the proposal text and slides, and writing and debugging the Python code and figure scripts. The author chose the topic from AI-suggested options and is responsible for the accuracy of the content. All experiments were executed with the included code, and every number in this document comes from its output (results.json); no results or references were invented.", B),
 Marker("refs")]

# references + appendix
refs = [
 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <i>Proc. 34th Int. Conf. Machine Learning (ICML)</i>, PMLR vol. 70, 2017, pp. 1321-1330.',
 'D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and perturbations," in <i>Proc. Int. Conf. Learning Representations (ICLR)</i>, 2019.',
 'Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in <i>Proc. 33rd ICML</i>, PMLR vol. 48, 2016, pp. 1050-1059.',
 'B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017.',
 'Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017.',
 'Y. Ovadia et al., "Can you trust your model\'s uncertainty? Evaluating predictive uncertainty under dataset shift," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 32, 2019.',
 'D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," in <i>Proc. ICLR</i>, 2017.',
 'A. Krizhevsky, "Learning multiple layers of features from tiny images," Tech. Rep., University of Toronto, 2009.',
 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011.',
 'D. Dua and C. Graff, "UCI Machine Learning Repository" (Optical Recognition of Handwritten Digits data set), University of California, Irvine, 2019. [Online]. Available: https://archive.ics.uci.edu',
 'OECD, <i>Frascati Manual 2015: Guidelines for Collecting and Reporting Data on Research and Experimental Development</i>. Paris: OECD Publishing, 2015.',
 'J. C. Mankins, "Technology readiness levels: A white paper," NASA, 1995.',
 'M. D. Wilkinson et al., "The FAIR Guiding Principles for scientific data management and stewardship," <i>Scientific Data</i>, vol. 3, Art. no. 160018, 2016.']
st += [PageBreak(), P("References", H)] + [P(f"[{i+1}] {t}", ParagraphStyle("R", parent=SM, leftIndent=18, firstLineIndent=-18, spaceAfter=2.5, alignment=TA_LEFT)) for i, t in enumerate(refs)]
# appendix tables
conds = ["clean"] + keys
ap = [["Condition"] + ["Acc." if False else "ECE " + n for n in ["soft", "TS-c", "TS-a", "MC", "Ens"]] + ["AURC " + n for n in ["soft", "TS-c", "TS-a", "MC", "Ens"]]]
for k in conds:
    ap.append([k] + [f3(ag(m, k, "ece")) for m in M] + [f3(ag(m, k, "aurc")) for m in M])
SMc = ParagraphStyle("SMc", parent=SM, fontSize=7, leading=8.4)
st += [P("Appendix A. Full results per condition", H),
 P("Mean over 10 splits. Condition names: noise/blur/fog followed by severity 1-3. Abbreviations: soft = softmax, TS-c = temperature scaling (clean), TS-a = temperature scaling (corrupted validation), MC = MC-dropout, Ens = ensemble of 5.", CAP),
 tbl(ap, [2.0 * cm] + [(W - 2.0 * cm) / 10] * 10, fs=SMc),
 P("Appendix B. Reproducibility", H),
 P("Code files: <font face='Courier'>cnn.py</font> (NumPy CNN, Adam, MC-dropout), <font face='Courier'>experiment.py</font> (corruptions, methods, metrics; writes results.json), <font face='Courier'>make_figures.py</font>, <font face='Courier'>demo.py</font> (video demo), <font face='Courier'>build_pdf.py</font>. "
   "Environment: Python 3.12, NumPy, SciPy, scikit-learn, Matplotlib; no GPU. Run: <font face='Courier'>python experiment.py</font> (about 80 s). Seeds 0-9; network seeds seed x 100 + member index. Repository: " + clink + ".", B),
 P("Appendix C. Demo output", H), img("demo_output.png", W * 0.72),
 P("Figure C1. Demo (demo.py): heavy synthetic fog (haze weight 0.7), ensemble of 5 CNNs, 20% least confident images rejected. Shown are the first 6 rejected and the first 6 accepted test images (not hand-picked). In this demo, accuracy rose from 63.8% (all images) to 70.0% (accepted images); 55 of 163 errors were rejected.", CAP)]
doc.build(st)
main_pages = Marker.pages["refs"]; print("proposal main pages (refs start on page after):", main_pages, " total:", len(PdfReader("proposal.pdf").pages))

# ------------------------------------------------------------------ slides ----
SW, SH = 960, 540
from reportlab.platypus import Paragraph as Pp
sc = rl_canvas.Canvas("slides.pdf", pagesize=(SW, SH)); sc.setTitle("CALIBRIX slides")
def para(c, text, x, ytop, w, size=19, lead=None, color=colors.black, bold=False, align=TA_LEFT):
    s = ParagraphStyle("s", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=lead or size * 1.28, textColor=color, alignment=align)
    p = Pp(text, s); w_, h_ = p.wrap(w, 1000); p.drawOn(c, x, ytop - h_); return h_
def frame(c, n, title):
    c.setFillColor(colors.white); c.rect(0, 0, SW, SH, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, 0, SW, 34, fill=1, stroke=0); c.setFillColor(colors.white); c.setFont("Helvetica", 11)
    c.drawString(30, 12, "CALIBRIX  |  Assignment 12  |  " + AUTHOR); c.drawRightString(SW - 30, 12, f"{n} / 7")
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 31); c.drawString(40, SH - 62, title); c.setFillColor(ORANGE); c.rect(40, SH - 74, 90, 4, fill=1, stroke=0)
def bullets(c, items, x, ytop, w, size=19, gap=9):
    y = ytop
    for it in items:
        c.setFillColor(ORANGE); c.circle(x + 5, y - size * 0.62, 3.6, fill=1, stroke=0)
        h = para(c, it, x + 20, y, w - 20, size); y -= h + gap
    return y
def pic(c, path, x, y, w, h):
    from PIL import Image as PI
    iw, ih = PI.open(path).size; k = min(w / iw, h / ih); c.drawImage(path, x + (w - iw * k) / 2, y + (h - ih * k) / 2, iw * k, ih * k, mask="auto")
def chip(c, text, x, y, w, h=30, fill=LIGHT, fs=14, tc=NAVY):
    c.setFillColor(fill); c.roundRect(x, y, w, h, 8, fill=1, stroke=0); c.setFillColor(tc); c.setFont("Helvetica-Bold", fs); c.drawCentredString(x + w / 2, y + h / 2 - fs * 0.35, text)

# 1 title
c = sc; c.setFillColor(NAVY); c.rect(0, 0, SW, SH, fill=1, stroke=0); c.setFillColor(ORANGE); c.rect(60, 330, 120, 5, fill=1, stroke=0)
c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 66); c.drawString(60, 400, "CALIBRIX")
para(c, "Teaching image classifiers to say <i>\"I don't know\"</i> in fog, noise and blur", 60, 310, 800, 32, color=colors.white)
para(c, "Mini R&amp;D project: applied research, TRL 3 now, TRL 4 target, 18 months", 60, 215, 800, 20, color=colors.HexColor("#d9dcec"))
para(c, f"{AUTHOR}  |  Research Tools and Methods, SDU University", 60, 170, 800, 18, color=colors.HexColor("#d9dcec"))
para(c, "Video: " + (VIDEO_URL or "[add link]"), 60, 135, 840, 15, color=ORANGE); sc.showPage()
# 2 problem
frame(sc, 2, "The problem and the gap")
bullets(sc, ["Outdoor cameras meet <b>fog, noise and blur</b>.", "Networks often stay <b>confident when they are wrong</b> [1], [6].", "A silent, confident error is worse than \"I don't know\".",
             "<b>Gap:</b> little per-corruption evidence on the <b>reject option</b>, and on what each method <b>costs</b>."], 40, SH - 100, 520, 21, 13)
sc.setFillColor(LIGHT); sc.roundRect(580, 150, 345, 290, 12, fill=1, stroke=0)
para(sc, "Type of R&amp;D: <b>applied research</b>", 600, 425, 310, 19, color=NAVY)
yy = 385
for t in ["Novel: new per-corruption evidence", "Creative: synthetic corruptions as calibration data", "Uncertain: it can fail (and partly did)", "Systematic: criteria set before running", "Reproducible: open code, 80 s run"]:
    sc.setFillColor(ORANGE); sc.circle(605, yy - 8, 3.8, fill=1, stroke=0); h = para(sc, t, 618, yy, 290, 16, color=colors.black); yy -= h + 10
para(sc, "Frascati criteria [11]", 600, 175, 290, 12, color=colors.HexColor("#666666")); sc.showPage()
# 3 approach
frame(sc, 3, "Approach and prototype")
pic(sc, "fig4_pipeline.png", 30, 300, 900, 170)
bullets(sc, ["<b>Methods:</b> softmax (baseline), temperature scaling (2 variants), MC-dropout, ensemble of 5 CNNs", "<b>Data:</b> 1,797 real digit images (8x8); noise, blur, fog at 3 severities; 10 random splits",
             "<b>Metrics:</b> ECE (calibration), AURC and accuracy at 80% coverage (reject option)", "<b>Target set before running:</b> ECE -30%, AURC -10% vs softmax"], 40, 280, 880, 20, 11); sc.showPage()
# 4 results
frame(sc, 4, "Prototype results (real runs)")
pic(sc, "fig1_bars.png", 20, 130, 640, 330)
chip(sc, f"Ensemble ECE {rel('ensemble','ece'):+.0f}%", 660, 380, 260, 44, fs=19); chip(sc, f"Ensemble AURC {rel('ensemble','aurc'):+.0f}%", 660, 320, 260, 44, fs=19)
para(sc, f"Mean over 9 corrupted conditions, 10 splits.<br/>Softmax: ECE {f3(S['softmax']['ece'][0])}, AURC {f4(S['softmax']['aurc'][0])}.<br/>Ensemble: ECE {f3(S['ensemble']['ece'][0])}, AURC {f4(S['ensemble']['aurc'][0])}.", 660, 295, 270, 15, color=colors.HexColor("#333333"))
para(sc, "Lower is better. Accuracy at 80%% coverage: %.1f%% (softmax) vs %.1f%% (ensemble)." % (100 * S['softmax']['acc80'][0], 100 * S['ensemble']['acc80'][0]), 40, 125, 880, 16, color=colors.HexColor("#333333")); sc.showPage()
# 5 honest findings + demo
frame(sc, 5, "What worked, what did not")
bullets(sc, [f"<b>Ensemble</b> is best, but the ECE target (-30%) was <b>missed</b>; the AURC gain is <b>not significant</b> (p = {a_p:.2f}).", f"<b>Temperature scaling on clean data</b> does not help under corruption (T = {Tc:.2f}).",
             "<b>MC-dropout</b> is inconsistent and costs 30x at inference.", "<b>Limits:</b> tiny 8x8 images, synthetic corruptions, small CNN."], 40, SH - 100, 880, 18, 7)
pic(sc, "demo_output.png", 30, 50, 640, 255)
para(sc, "<b>Live demo:</b> heavy fog, the ensemble rejects the 20% least confident images. Accuracy goes from 63.8% (all) to 70.0% (accepted).", 690, 290, 240, 16, color=colors.HexColor("#333333")); sc.showPage()
# 6 plan
frame(sc, 6, "Work plan, budget and risks")
pic(sc, "fig3_gantt.png", 30, 245, 900, 225)
chip(sc, f"Budget: {total:,} thousand KZT (about {total/1000:.1f} M)", 40, 200, 420, 34, fs=15); chip(sc, "5 work packages, 4 milestones, go/no-go at month 14", 480, 200, 440, 34, fs=15)
bullets(sc, ["<b>Top risks:</b> results do not transfer to CIFAR-10-C (go/no-go M3, report negative results); GPU delay (order in month 1, cloud backup)", "<b>Ethics and data:</b> no personal data; FAIR repository with DOI; code under MIT licence"], 40, 175, 880, 17, 8); sc.showPage()
# 7 impact
frame(sc, 7, "Impact and next steps")
bullets(sc, ["<b>Result:</b> open benchmark, toolkit, cost report, at least one paper", "<b>Impact:</b> vision systems that flag uncertain images for a human instead of failing silently", "<b>Transfer:</b> open-source release, pilot with a partner, commercialisation grant (TRL 5-6)",
             "<b>Next step (TRL 4):</b> CIFAR-10-C data, ResNet-type CNNs, real fog images, cost measurements", "<b>Ask:</b> %.1f million KZT for 18 months" % (total / 1000)], 40, SH - 100, 880, 20, 14)
sc.setFillColor(NAVY); sc.roundRect(40, 60, 880, 80, 12, fill=1, stroke=0)
para(sc, "A classifier that knows when it does not know is safer to use.", 60, 112, 840, 24, color=colors.white, bold=True, align=TA_CENTER); sc.showPage(); sc.save()

# ------------------------------------------------------------------ merge ----
w = PdfWriter()
for f in ("proposal.pdf", "slides.pdf"):
    for pg in PdfReader(f).pages: w.add_page(pg)
w.add_metadata({"/Title": "Assignment 12 - CALIBRIX", "/Author": AUTHOR}); w.write(OUT)
print("slides pages:", len(PdfReader("slides.pdf").pages), " final pages:", len(PdfReader(OUT).pages), " budget total:", total)
