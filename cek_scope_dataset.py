
from pathlib import Path
import pandas as pd

# Lokasi dataset hasil penggabungan
DATASET_PATH = Path("dataset_v2/dataset_final.csv")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH.resolve()}")

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8-sig",
    keep_default_na=False
)

print("=" * 65)
print("PEMERIKSAAN DATASET CIVA")
print("=" * 65)

print(f"\nTotal baris       : {len(df):,}")
print(f"Kolom tersedia    : {list(df.columns)}")

# Periksa kolom wajib untuk evaluasi
required = ["judul", "author", "status", "jenis", "split", "scope"]
missing = [col for col in required if col not in df.columns]

if missing:
    print(f"\nKolom yang belum tersedia: {missing}")
    raise SystemExit("Evaluasi dihentikan. Periksa struktur dataset terlebih dahulu.")

# Distribusi scope
print("\n[1] DISTRIBUSI SCOPE")
print(df["scope"].replace("", "(kosong)").value_counts().to_string())

# Distribusi status
print("\n[2] DISTRIBUSI STATUS")
print(
    df["status"].value_counts()
    .rename(index={0: "Halu", 1: "Valid"})
    .to_string()
)

# Distribusi split
print("\n[3] DISTRIBUSI SPLIT")
print(df["split"].value_counts().to_string())

# Distribusi jenis berdasarkan scope
print("\n[4] DISTRIBUSI JENIS BERDASARKAN SCOPE")
print(
    pd.crosstab(
        df["scope"].replace("", "(kosong)"),
        df["jenis"],
        margins=True
    ).to_string()
)

# Distribusi status berdasarkan scope
print("\n[5] DISTRIBUSI STATUS BERDASARKAN SCOPE")
print(
    pd.crosstab(
        df["scope"].replace("", "(kosong)"),
        df["status"].replace({"0": "Halu", "1": "Valid", 0: "Halu", 1: "Valid"}),
        margins=True
    ).to_string()
)

print("\nPemeriksaan selesai.")
