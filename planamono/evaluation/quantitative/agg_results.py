import pandas as pd
from pathlib import Path

# ROOT = Path("scannetpp/eval")
ROOT = Path("")

METHODS = {
    "planercnn": "PlaneRCNN",
    "zeroplane": "ZeroPlane",
    "monobase": "MonoBase",
    "moge_ours": "Ours",
    "moge_ours_merged": "Ours_merged",
    
    "gtplanarity_ourseg": "gtplanarity_ourseg",
    "gtseg": "gtseg",
    "ourplanarity_gtseg": "ourplanarity_gtseg",
}

def find_dataset_csv(folder: Path):
    files = list(folder.glob("*_results_dataset.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No *_results_dataset.csv in {folder}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple dataset CSVs in {folder}: {files}")
    return files[0]

# ---------- TABLE 1: Precision / Recall ----------
rows_pr = []

for folder_name, method_name in METHODS.items():
    folder = ROOT / folder_name
    if not folder.exists():
        print(f"[WARN] Missing folder {folder}")
        continue

    csv_path = find_dataset_csv(folder)
    df = pd.read_csv(csv_path).iloc[0]

    rows_pr.append({
        "Method": method_name,
        "num_scenes": int(df["num_scenes"]),
        "num_frames": int(df["num_frames_total"]),
        "prec@1cm": df["prec@1cm_mean"],
        "recall@1cm": df["rec@1cm_mean"],
        "prec@2cm": df["prec@2cm_mean"],
        "recall@2cm": df["rec@2cm_mean"],
        "prec@5cm": df["prec@5cm_mean"],
        "recall@5cm": df["rec@5cm_mean"],
    })

df_pr = pd.DataFrame(rows_pr)
df_pr.to_csv("table_precision_recall_baseline.csv", index=False)

# ---------- TABLE 2: Segmentation ----------
rows_seg = []

for folder_name, method_name in METHODS.items():
    folder = ROOT / folder_name
    if not folder.exists():
        continue

    csv_path = find_dataset_csv(folder)
    df = pd.read_csv(csv_path).iloc[0]

    rows_seg.append({
        "Method": method_name,
        "num_scenes": int(df["num_scenes"]),
        "num_frames": int(df["num_frames_total"]),
        "Rand Index": df["rand_index_mean"],
        "VOI": df["voi_mean"],
        "SC": df["sc_mean"],
    })

df_seg = pd.DataFrame(rows_seg)
df_seg.to_csv("table_segmentation_baseline.csv", index=False)

print("✅ Saved:")
print(" - table_precision_recall_baseline.csv")
print(" - table_segmentation_baseline.csv")