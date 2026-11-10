
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
BOOK_DIR = ROOT / "dataset" / "additional_books"

VALID_FILE = BOOK_DIR / "TAMBAHAN_data_buku_valid_5394.csv"
MISMATCH_FILE = BOOK_DIR / "TAMBAHAN_mismatch_buku_2kolom.csv"

sys.path.insert(0, str(ROOT))
from civa_core import clean_title, clean_authors

# 1. Baca dataset
valid = pd.read_csv(VALID_FILE, keep_default_na=False)
mismatch = pd.read_csv(MISMATCH_FILE, keep_default_na=False)

# 2. Normalisasi judul dan penulis dengan fungsi CIVA
valid["title_key"] = valid["judul"].map(clean_title)
valid["author_key"] = valid["author"].map(clean_authors)

mismatch["title_key"] = mismatch["judul"].map(clean_title)
mismatch["author_key"] = mismatch["author"].map(clean_authors)

# 3. Buat indeks judul valid
title_index = {}
for idx, row in valid.iterrows():
    title_index.setdefault(row["title_key"], []).append(idx)

# 4. Petakan mismatch berdasarkan judul yang sama
results = []

for idx, row in mismatch.iterrows():
    candidates = title_index.get(row["title_key"], [])

    if not candidates:
        results.append({
            "mismatch_id": idx,
            "judul_mismatch": row["judul"],
            "author_mismatch": row["author"],
            "jumlah_kandidat": 0,
            "valid_id": "",
            "judul_valid": "",
            "author_valid": "",
            "author_sama": False,
            "status_pemetaan": "TIDAK_DITEMUKAN_JUDUL_SAMA"
        })
        continue

    for candidate_idx in candidates:
        source = valid.loc[candidate_idx]
        same_author = row["author_key"] == source["author_key"]

        results.append({
            "mismatch_id": idx,
            "judul_mismatch": row["judul"],
            "author_mismatch": row["author"],
            "jumlah_kandidat": len(candidates),
            "valid_id": source["id"],
            "judul_valid": source["judul"],
            "author_valid": source["author"],
            "author_sama": same_author,
            "status_pemetaan": (
                "PASANGAN_SAMA_PERSIS"
                if same_author else "JUDUL_SAMA_PENULIS_BERBEDA"
            )
        })

# 5. Simpan hasil audit
result_df = pd.DataFrame(results)

output = BOOK_DIR / "audit_mismatch_buku.csv"
result_df.to_csv(output, index=False, encoding="utf-8-sig")

# 6. Ringkasan
print("\n========== HASIL AUDIT BUKU ==========")
print(f"Jumlah buku valid       : {len(valid)}")
print(f"Jumlah mismatch        : {len(mismatch)}")
print(f"Baris hasil pemetaan   : {len(result_df)}")

print("\nDistribusi status pemetaan:")
print(result_df["status_pemetaan"].value_counts().to_string())

print("\nJumlah mismatch tanpa judul sama:")
print(
    (result_df["status_pemetaan"] == "TIDAK_DITEMUKAN_JUDUL_SAMA")
    .sum()
)

print("\nFile hasil:")
print(output)
