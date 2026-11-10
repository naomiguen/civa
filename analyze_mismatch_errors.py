
from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("reports/prediksi_data_uji_buku.csv")
OUTPUT = Path("reports/analisis_kesalahan_mismatch.csv")

df = pd.read_csv(INPUT, encoding="utf-8-sig")

mismatch = df[df["jenis"] == "mismatch_buku"].copy()

if mismatch.empty:
    raise ValueError("Tidak ditemukan data mismatch_buku.")

# Prediksi 0 = halu terdeteksi
# Prediksi 1 = halu lolos sebagai valid
mismatch["hasil"] = np.where(
    mismatch["prediksi"] == 0,
    "Terdeteksi",
    "Lolos"
)

print("=" * 75)
print("ANALISIS KESALAHAN MISMATCH BUKU")
print("=" * 75)

print(f"\nTotal mismatch: {len(mismatch)}")

# 1. Analisis berdasarkan metode mapping
if "mapping_method" in mismatch.columns:
    print("\n[1] HASIL BERDASARKAN MAPPING METHOD")

    print(
        pd.crosstab(
            mismatch["mapping_method"],
            mismatch["hasil"],
            margins=True
        ).to_string()
    )

# 2. Analisis berdasarkan scope
print("\n[2] HASIL BERDASARKAN SCOPE")

print(
    pd.crosstab(
        mismatch["scope"],
        mismatch["hasil"],
        margins=True
    ).to_string()
)

# 3. Analisis similarity
similarity_columns = [
    "title_similarity",
    "author_similarity"
]

for col in similarity_columns:
    if col not in mismatch.columns:
        continue

    mismatch[col] = pd.to_numeric(
        mismatch[col],
        errors="coerce"
    )

    print(f"\n[3] DISTRIBUSI {col.upper()}")

    print(
        mismatch.groupby("hasil")[col]
        .agg(["count", "mean", "median", "min", "max"])
        .round(4)
        .to_string()
    )

# 4. Kesalahan yang perlu diperiksa
errors = mismatch[
    mismatch["hasil"] == "Lolos"
].copy()

errors.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n[4] JUMLAH KESALAHAN")
print(f"Mismatch terdeteksi : {(mismatch['hasil'] == 'Terdeteksi').sum()}")
print(f"Mismatch lolos      : {len(errors)}")

print(f"\nFile kesalahan disimpan di: {OUTPUT}")
