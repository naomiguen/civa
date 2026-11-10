
from pathlib import Path
import pandas as pd

INPUT = Path("reports/analisis_kesalahan_mismatch.csv")

df = pd.read_csv(INPUT, encoding="utf-8-sig")

# Tampilkan kolom yang relevan untuk audit manual
columns = [
    "judul",
    "author",
    "judul_asal",
    "author_asal",
    "scope",
    "mapping_method",
    "title_similarity",
    "author_similarity",
    "p_valid",
    "prediksi",
]

available = [col for col in columns if col in df.columns]

# Prioritaskan data dengan author similarity rendah
df["author_similarity"] = pd.to_numeric(
    df["author_similarity"],
    errors="coerce"
)

df = df.sort_values(
    by=["author_similarity", "title_similarity"],
    ascending=[True, False]
)

print("=" * 85)
print("AUDIT MANUAL MISMATCH BUKU YANG LOLOS")
print("=" * 85)

print(f"Total data: {len(df)}")

print("\n10 contoh dengan author similarity terendah:")
print(
    df[available]
    .head(10)
    .to_string(index=False)
)

print("\n10 contoh dengan author similarity tertinggi:")
print(
    df[available]
    .tail(10)
    .to_string(index=False)
)

output = Path("reports/contoh_mismatch_lolos_audit.csv")

df[available].to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nFile disimpan: {output}")
