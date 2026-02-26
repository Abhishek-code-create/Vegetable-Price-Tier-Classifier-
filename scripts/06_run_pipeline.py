

import subprocess, sys, os, time

STEPS = [
    ("scripts/01_preprocess.py",    "Data Preprocessing"),
    ("scripts/02_eda.py",           "Exploratory Data Analysis"),
    ("scripts/03_train_evaluate.py","Classification Model Training"),
    ("scripts/04_clustering.py",    "K-Means Clustering"),
]

print("=" * 65)
print("  STW5000CEM — Vegetable Price Tier Classifier")
print("  Full Pipeline Runner")
print("=" * 65)

total_start = time.time()
for script, label in STEPS:
    print(f"\n{'─'*65}\n  ▶  {label}\n{'─'*65}")
    t = time.time()
    r = subprocess.run([sys.executable, script], capture_output=False)
    if r.returncode != 0:
        print(f"\n   FAILED: {label}")
        sys.exit(1)
    print(f"\n  ✅ Done  ({time.time()-t:.1f}s)")

print(f"\n{'='*65}")
print(f"  ✅ Full pipeline complete in {time.time()-total_start:.1f}s")
print(f"{'='*65}")
print("\n  To classify a vegetable price tier:")
print("    python scripts/05_cli.py")
print("    python scripts/05_cli.py --vegetable 'Cauli Local' --date 2024-06-01")
print("\n  Open outputs/VegetablePriceClassifier_App.html in a browser for the GUI.")
