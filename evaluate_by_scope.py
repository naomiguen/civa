
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import civa_core
from civa_core import clean_title, clean_authors


# ============================================================
# KONFIGURASI
# ============================================================

DATASET_PATH = Path("dataset_v2/dataset_final.csv")
THRESHOLD = 0.5

# ============================================================
# 1. MEMUAT DATASET
# ============================================================

print("=" * 75)
print("EVALUASI CIVA BERDASARKAN SCOPE BUKU")
print("=" * 75)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_PATH.resolve()}"
    )

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8-sig",
    keep_default_na=False
)

required_columns = [
    "judul", "author", "status", "jenis",
    "grup_id", "split", "scope"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

df["status"] = df["status"].astype(int)
df["judul"] = df["judul"].astype(str).str.strip()
df["author"] = df["author"].astype(str).str.strip()

print(f"\nJumlah data awal: {len(df):,}")

# ============================================================
# 2. PEMBERSIHAN DATA
# Mengikuti aturan load_dataset() pada train_model.py.
# Tidak melakukan augmentasi inisial pada data evaluasi.
# ============================================================

title_key = df["judul"].map(clean_title)
author_key = df["author"].map(clean_authors)

usable = (
    (title_key.str.len() >= 4)
    & ~((df["author"] != "") & (author_key == ""))
)

pair = title_key + "\x1f" + author_key

is_halu = df["status"] == 0

contradictory = (
    is_halu
    & pair.isin(set(pair[~is_halu]))
)

keep = usable & ~contradictory

duplicate = keep & pair.where(keep).duplicated()
keep &= ~duplicate

print(f"Data tidak terpakai : {(~usable).sum():,}")
print(f"Halu identik valid  : {contradictory.sum():,}")
print(f"Duplikat            : {duplicate.sum():,}")

df = df.loc[keep].copy().reset_index(drop=True)

print(f"Jumlah setelah bersih: {len(df):,}")

# ============================================================
# 3. MEMILIH DATA UJI BUKU
# ============================================================

test = df[
    (df["split"] == "uji")
    & (df["scope"].isin(["cs", "non_cs"]))
].copy().reset_index(drop=True)

if test.empty:
    raise ValueError("Data uji buku dengan scope CS/non-CS kosong.")

print("\nDistribusi data uji:")
print(
    pd.crosstab(
        test["scope"],
        test["status"].map({0: "Halu", 1: "Valid"}),
        margins=True
    ).to_string()
)

# ============================================================
# 4. MEMUAT MODEL TERSIMPAN
# ============================================================

print("\nMemuat model CIVA...")

model = civa_core.load_artifacts()

X_test = test[["judul", "author"]]

print("Melakukan prediksi...")
p_valid = model.predict_proba(X_test)[:, 1]

test["p_valid"] = p_valid
test["prediksi"] = (p_valid >= THRESHOLD).astype(int)

print("Prediksi selesai.")

# ============================================================
# 5. FUNGSI EVALUASI
# ============================================================

def evaluate_group(group, group_name):
    y_true = group["status"].to_numpy()
    probabilities = group["p_valid"].to_numpy()
    y_pred = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "kelompok": group_name,
        "jumlah": len(group),
        "valid": int((y_true == 1).sum()),
        "halu": int((y_true == 0).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_halu": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "recall_halu": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "f1_halu": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "precision_valid": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_valid": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_valid": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "halu_terdeteksi": int(tn),
        "halu_lolos": int(fp),
        "valid_salah_dituduh_halu": int(fn),
        "valid_dikenali": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true, probabilities
        )
        result["ap_halu"] = average_precision_score(
            1 - y_true, 1 - probabilities
        )
    else:
        result["roc_auc"] = np.nan
        result["ap_halu"] = np.nan

    return result


# ============================================================
# 6. EVALUASI CS DAN NON-CS
# ============================================================

results = []

for scope in ["cs", "non_cs"]:
    group = test[test["scope"] == scope]

    if group.empty:
        print(f"\nPeringatan: data uji {scope} kosong.")
        continue

    results.append(evaluate_group(group, scope.upper()))


# ============================================================
# 7. EVALUASI KHUSUS MISMATCH BUKU
# ============================================================

mismatch = test[test["jenis"] == "mismatch_buku"]

for scope in ["cs", "non_cs"]:
    group = mismatch[mismatch["scope"] == scope]

    if group.empty:
        continue

    detected = int((group["p_valid"] < THRESHOLD).sum())
    missed = len(group) - detected

    print(f"\nMISMATCH BUKU - {scope.upper()}")
    print(f"Jumlah data uji       : {len(group)}")
    print(f"Berhasil terdeteksi   : {detected}")
    print(f"Lolos sebagai valid   : {missed}")
    print(
        f"Tingkat deteksi       : "
        f"{detected / len(group) * 100:.2f}%"
    )


# ============================================================
# 8. MENAMPILKAN HASIL PER SCOPE
# ============================================================

result_df = pd.DataFrame(results)

if not result_df.empty:
    metric_columns = [
        "kelompok", "jumlah", "valid", "halu",
        "roc_auc", "ap_halu",
        "accuracy", "balanced_accuracy",
        "precision_halu", "recall_halu", "f1_halu",
        "precision_valid", "recall_valid", "f1_valid",
        "f1_macro",
        "halu_terdeteksi", "halu_lolos",
        "valid_salah_dituduh_halu", "valid_dikenali"
    ]

    print("\n" + "=" * 110)
    print("HASIL EVALUASI PER SCOPE")
    print("=" * 110)

    print(
        result_df[metric_columns]
        .round(4)
        .to_string(index=False)
    )

# ============================================================
# 9. MENYIMPAN HASIL
# ============================================================

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

result_df.to_csv(
    output_dir / "evaluasi_per_scope.csv",
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    output_dir / "prediksi_data_uji_buku.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFile berhasil disimpan:")
print("- reports/evaluasi_per_scope.csv")
print("- reports/prediksi_data_uji_buku.csv")

print("\nEvaluasi selesai.")
