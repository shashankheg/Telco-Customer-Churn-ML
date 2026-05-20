import os
import glob
import shutil
import json


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPORT_DIR = os.path.join(PROJECT_ROOT, "models_export","model")


# Find latest trained model in mlruns (exclude metadata subfolders)
paths = [
    p for p in glob.glob(os.path.join(PROJECT_ROOT, "mlruns", "*", "*", "artifacts", "model"))
    if os.path.exists(os.path.join(p, "MLmodel"))  # must have MLmodel file
    and "metadata" not in p                         # exclude metadata subfolders
]

if not paths:
    raise Exception("No model found. Run the pipeline first.")

latest = max(paths, key=os.path.getctime)
run_dir = os.path.join(os.path.dirname(latest), "..")

print(f"Exporting model from run: {latest}")

#clearn and recrteate export directory
shutil.rmtree(os.path.join(PROJECT_ROOT, "models_export"), ignore_errors=True)
os.makedirs(EXPORT_DIR)


# Copy MLflow model artifacts

for item in os.listdir(latest):
    src = os.path.join(latest, item)
    dst = os.path.join(EXPORT_DIR, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


#Copy feature files from the artifacts directory.
artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")

for fname in ["feature_columns.json", "feature_columns.txt", "preprocessing.pkl"]:
    src = os.path.join(artifacts_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(EXPORT_DIR, fname))
        print(f"✅ Copied {fname}")

print(f"\n✅ Exported to: {EXPORT_DIR}")
print(f"   Files: {os.listdir(EXPORT_DIR)}")
print(f"\nNow run: docker build -t telco-churn . --no-cache")

