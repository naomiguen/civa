
from pathlib import Path
from difflib import SequenceMatcher
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
BOOK_DIR = DATASET_DIR / "additional_books"
OUTPUT_DIR = ROOT / "dataset_v2"

OLD_FILE = DATASET_DIR / "dataset_final.csv"
VALID_FILE = BOOK_DIR / "TAMBAHAN_data_buku_valid_5394.csv"
MISMATCH_FILE = BOOK_DIR / "TAMBAHAN_mismatch_buku_2kolom.csv"
FUZZY_FILE = BOOK_DIR / "audit_mismatch_fuzzy.csv"

SEED = 42
TEST_SIZE = 0.20


def normalize_title(value):
    from civa_core import clean_title
    return clean_title(str(value))


def normalize_author(value):
    from civa_core import clean_authors
    return clean_authors(str(value))


def author_similarity(a, b):
    a = normalize_author(a)
    b = normalize_author(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def main():
    sys.path.insert(0, str(ROOT))

    print("=" * 65)
    print("CIVA - PENGGABUNGAN DATASET BUKU")
    print("=" * 65)

    # --------------------------------------------------
    # 1. BACA DATASET
    # --------------------------------------------------
    print("\n[1/7] Membaca dataset...")

    old = pd.read_csv(
        OLD_FILE, encoding="utf-8-sig", keep_default_na=False
    )
    valid = pd.read_csv(
        VALID_FILE, encoding="utf-8-sig", keep_default_na=False
    )
    mismatch = pd.read_csv(
        MISMATCH_FILE, encoding="utf-8-sig", keep_default_na=False
    )
    fuzzy = pd.read_csv(
        FUZZY_FILE, encoding="utf-8-sig", keep_default_na=False
    )

    required_old = {
        "judul", "author", "status", "jenis", "grup_id", "split"
    }
    required_valid = {
        "id", "judul", "author", "label_asli", "scope", "bidang", "bahasa"
    }
    required_mismatch = {"judul", "author"}
    required_fuzzy = {
        "mismatch_id", "rank", "valid_id", "title_similarity"
    }

    for name, frame, required in [
        ("dataset lama", old, required_old),
        ("dataset valid buku", valid, required_valid),
        ("dataset mismatch", mismatch, required_mismatch),
        ("audit fuzzy", fuzzy, required_fuzzy),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan pada {name}: {sorted(missing)}"
            )

    print(f"Dataset lama    : {len(old):,}")
    print(f"Buku valid      : {len(valid):,}")
    print(f"Buku mismatch   : {len(mismatch):,}")

    # --------------------------------------------------
    # 2. NORMALISASI
    # --------------------------------------------------
    print("\n[2/7] Normalisasi judul dan penulis...")

    valid["title_key"] = valid["judul"].map(normalize_title)
    valid["author_key"] = valid["author"].map(normalize_author)

    mismatch["title_key"] = mismatch["judul"].map(normalize_title)
    mismatch["author_key"] = mismatch["author"].map(normalize_author)

    valid["id"] = valid["id"].astype(int)

    if valid["id"].duplicated().any():
        raise ValueError("ID buku valid tidak unik.")

    # --------------------------------------------------
    # 3. BENTUK PEMETAAN SUMBER MISMATCH
    # --------------------------------------------------
    print("\n[3/7] Memetakan mismatch ke buku valid...")

    # Indeks buku valid berdasarkan judul ternormalisasi
    title_index = {}

    for idx, row in valid.iterrows():
        title_index.setdefault(row["title_key"], []).append(idx)

    mappings = {}

    # 3A. Utamakan judul yang sama setelah normalisasi.
    for mismatch_id, row in mismatch.iterrows():
        candidates = title_index.get(row["title_key"], [])

        if not candidates:
            continue

        # Jika ada beberapa judul sama, pilih penulis paling mirip.
        best_idx = max(
            candidates,
            key=lambda idx: author_similarity(
                row["author_key"], valid.loc[idx, "author_key"]
            )
        )

        source = valid.loc[best_idx]

        mappings[mismatch_id] = {
            "valid_id": int(source["id"]),
            "method": "exact_normalized_title",
            "title_similarity": 1.0,
            "author_similarity": author_similarity(
                row["author_key"], source["author_key"]
            )
        }

    exact_count = len(mappings)

    # 3B. Untuk judul yang tidak sama, gunakan fuzzy rank 1.
    fuzzy_top = fuzzy[fuzzy["rank"].astype(int) == 1].copy()

    fuzzy_top["mismatch_id"] = fuzzy_top["mismatch_id"].astype(int)
    fuzzy_top["valid_id"] = fuzzy_top["valid_id"].astype(int)

    for _, candidate in fuzzy_top.iterrows():
        mismatch_id = int(candidate["mismatch_id"])

        if mismatch_id in mappings:
            continue

        source_rows = valid[valid["id"] == int(candidate["valid_id"])]

        if source_rows.empty:
            raise ValueError(
                f"Valid ID {candidate['valid_id']} tidak ditemukan."
            )

        source = source_rows.iloc[0]
        original = mismatch.iloc[mismatch_id]

        mappings[mismatch_id] = {
            "valid_id": int(source["id"]),
            "method": "fuzzy_rank_1",
            "title_similarity": float(candidate["title_similarity"]),
            "author_similarity": author_similarity(
                original["author_key"], source["author_key"]
            )
        }

    if len(mappings) != len(mismatch):
        missing_ids = sorted(set(range(len(mismatch))) - set(mappings))
        raise ValueError(
            f"Ada mismatch yang belum memiliki sumber: {missing_ids[:20]}"
        )

    print(f"Pemetaan judul sama : {exact_count:,}")
    print(f"Pemetaan fuzzy      : {len(mappings) - exact_count:,}")
    print(f"Total terpetakan    : {len(mappings):,}")

    # --------------------------------------------------
    # 4. PEMBAGIAN GROUP TRAIN / TEST
    # --------------------------------------------------
    print("\n[4/7] Membuat pembagian group train-test...")

    old_groups = set(old["grup_id"].astype(int).unique())
    book_ids = valid["id"].astype(int).tolist()

    if old_groups.intersection(book_ids):
        raise ValueError(
            "ID buku bertabrakan dengan grup_id dataset lama."
        )

    groups_with_halu = {
        info["valid_id"] for info in mappings.values()
    }

    group_table = pd.DataFrame({
        "grup_id": book_ids,
        "has_halu": [
            int(book_id in groups_with_halu)
            for book_id in book_ids
        ]
    })

    train_groups, test_groups = train_test_split(
        group_table["grup_id"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=group_table["has_halu"]
    )

    split_map = {
        **{int(g): "latih" for g in train_groups},
        **{int(g): "uji" for g in test_groups}
    }

    # --------------------------------------------------
    # 5. BENTUK BARIS BUKU VALID
    # --------------------------------------------------
    print("\n[5/7] Menyusun baris buku valid dan mismatch...")

    book_rows = []

    for _, row in valid.iterrows():
        book_id = int(row["id"])

        book_rows.append({
            "judul": row["judul"],
            "author": row["author"],
            "tahun": row.get("tahun", ""),
            "label": "valid",
            "status": 1,
            "jenis": "asli",
            "grup_id": book_id,
            "split": split_map[book_id],
            "sumber_data": "OpenLibrary",
            "bidang": row.get("bidang", ""),
            "bahasa": row.get("bahasa", ""),
            "judul_asal": "",
            "author_asal": "",
            "scope": row.get("scope", ""),
            "jenis_referensi": row.get("jenis_referensi", "book"),
            "link_referensi": row.get("link_referensi", ""),
            "mapping_method": "source_valid",
            "title_similarity": 1.0,
            "author_similarity": 1.0
        })

    # --------------------------------------------------
    # 6. BENTUK BARIS MISMATCH
    # --------------------------------------------------
    for mismatch_id, row in mismatch.iterrows():
        info = mappings[mismatch_id]
        source = valid.loc[valid["id"] == info["valid_id"]].iloc[0]

        book_rows.append({
            "judul": row["judul"],
            "author": row["author"],
            "tahun": source.get("tahun", ""),
            "label": "halu",
            "status": 0,
            "jenis": "mismatch_buku",
            "grup_id": info["valid_id"],
            "split": split_map[info["valid_id"]],
            "sumber_data": "OpenLibrary_mismatch",
            "bidang": source.get("bidang", ""),
            "bahasa": source.get("bahasa", ""),
            "judul_asal": source["judul"],
            "author_asal": source["author"],
            "scope": source.get("scope", ""),
            "jenis_referensi": "book",
            "link_referensi": source.get("link_referensi", ""),
            "mapping_method": info["method"],
            "title_similarity": info["title_similarity"],
            "author_similarity": info["author_similarity"]
        })

    books = pd.DataFrame(book_rows)

    # --------------------------------------------------
    # 7. GABUNGKAN DAN SIMPAN KE FOLDER BARU
    # --------------------------------------------------
    print("\n[6/7] Menggabungkan dengan dataset lama...")

    # Dataset lama tidak diubah; kolom tambahan diisi kosong.
    for col in books.columns:
        if col not in old.columns:
            old[col] = ""

    for col in old.columns:
        if col not in books.columns:
            books[col] = ""

    # ID baris baru dibuat unik.
    max_id = (
        pd.to_numeric(old["id_baris"], errors="coerce").max()
        if "id_baris" in old.columns else len(old) - 1
    )
    start_id = int(max_id) + 1 if pd.notna(max_id) else len(old)

    books["id_baris"] = range(
    start_id,
    start_id + len(books)
)

    # Susun kolom mengikuti dataset lama, lalu metadata tambahan.
    columns = list(old.columns) + [
        c for c in books.columns if c not in old.columns
    ]

    combined = pd.concat(
        [old[columns], books[columns]],
        ignore_index=True
    )

    combined["status"] = pd.to_numeric(
        combined["status"], errors="raise"
    ).astype(int)

    combined["grup_id"] = pd.to_numeric(
        combined["grup_id"], errors="raise"
    ).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_path = OUTPUT_DIR / "dataset_final.csv"
    halu_path = OUTPUT_DIR / "dataset_halu.csv"
    mapping_path = OUTPUT_DIR / "mapping_mismatch_buku.csv"

    combined.to_csv(
        final_path, index=False, encoding="utf-8-sig"
    )

    combined[combined["status"] == 0].to_csv(
        halu_path, index=False, encoding="utf-8-sig"
    )

    mapping_rows = []

    for mismatch_id, info in mappings.items():
        mapping_rows.append({
            "mismatch_id": mismatch_id,
            "valid_id": info["valid_id"],
            "mapping_method": info["method"],
            "title_similarity": info["title_similarity"],
            "author_similarity": info["author_similarity"],
            "split": split_map[info["valid_id"]]
        })

    pd.DataFrame(mapping_rows).to_csv(
        mapping_path, index=False, encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # VALIDASI AKHIR
    # --------------------------------------------------
    print("\n[7/7] Validasi hasil penggabungan...")

    split_conflicts = (
        combined.groupby("grup_id")["split"].nunique() > 1
    ).sum()

    print("\n========== RINGKASAN DATASET V2 ==========")
    print(f"Total baris             : {len(combined):,}")
    print(f"Total valid             : {(combined.status == 1).sum():,}")
    print(f"Total halu              : {(combined.status == 0).sum():,}")
    print(f"Jumlah grup             : {combined.grup_id.nunique():,}")
    print(f"Split latih             : {(combined.split == 'latih').sum():,}")
    print(f"Split uji               : {(combined.split == 'uji').sum():,}")
    print(f"Konflik split per grup  : {split_conflicts}")

    print("\nDistribusi jenis:")
    print(combined["jenis"].value_counts().to_string())

    print("\nDistribusi scope buku:")
    print(
        books[books["status"] == 1]["scope"]
        .value_counts(dropna=False).to_string()
    )

    if split_conflicts != 0:
        raise ValueError("Ditemukan group yang tersebar pada dua split.")

    if len(combined) != len(old) + len(valid) + len(mismatch):
        raise ValueError("Jumlah baris hasil gabungan tidak sesuai.")

    print("\nFile berhasil dibuat:")
    print(final_path)
    print(halu_path)
    print(mapping_path)

    print("\nDataset lama tetap tidak berubah.")
    print("Belum ada training yang dijalankan.")


if __name__ == "__main__":
    main()
