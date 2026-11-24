# CALIBRIX prototype code
Pure NumPy + scikit-learn. No GPU. Python 3.12.

    pip install numpy scipy scikit-learn matplotlib reportlab pypdf pillow
    python gradcheck.py       # numerical gradient check of the CNN
    python experiment.py      # ~80 s, writes results.json (all numbers in the report)
    python make_figures.py    # figures
    python demo.py            # 10 s demo for the video
    python build_pdf.py       # builds the report PDF (edit AUTHOR, VIDEO_URL, CODE_URL at the top)

Files: cnn.py (NumPy CNN), experiment_lib.py (corruptions), experiment.py (main study), results.json (output).
