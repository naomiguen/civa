
from pathlib import Path
import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
BOOK_DIR = ROOT / "dataset" / "additional_books"

VALID_FILE = BOOK_DIR / "TAMBAHAN_data_buku_valid_5394.csv"
MISMATCH_FILE = BOOK_DIR / "TAMBAHAN_mismatch_buku_2kolom.csv"
EXACT_AUDIT_FILE = BOOK_DIR / "audit_mismatch_buku.csv"

sys.path.insert(0, str(ROOT))
from civa_core import clean_title, clean_authors


def main():
    print("Membaca dataset...")

    valid = pd.read_csv(VALID_FILE, keep_default_na=False)
    mismatch = pd.read_csv(MISMATCH_FILE, keep_default_na=False)
    exact_audit = pd.read_csv(EXACT_AUDIT_FILE, keep_default_na=False)

    # Ambil mismatch yang belum memiliki judul sama persis
    unmatched_ids = sorted(
        set(
            exact_audit.loc[
                exact_audit["status_pemetaan"]
                == "TIDAK_DITEMUKAN_JUDUL_SAMA",
                "mismatch_id"
            ].astype(int)
        )
    )

    print(f"Jumlah buku valid       : {len(valid)}")
    print(f"Mismatch untuk diproses : {len(unmatched_ids)}")

    # Normalisasi menggunakan fungsi asli CIVA
    print("\nNormalisasi judul dan penulis...")

    valid["title_key"] = valid["judul"].map(clean_title)
    valid["author_key"] = valid["author"].map(clean_authors)

    mismatch["title_key"] = mismatch["judul"].map(clean_title)
    mismatch["author_key"] = mismatch["author"].map(clean_authors)

    # Hapus judul kosong dari kandidat
    valid = valid[
        valid["title_key"].astype(str).str.strip() != ""
    ].reset_index(drop=True)

    queries = (
        mismatch.loc[unmatched_ids, "title_key"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    references = (
        valid["title_key"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nMembentuk TF-IDF character n-gram...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=True,
        min_df=1,
        dtype="float32"
    )

    combined = references + queries
    matrix = vectorizer.fit_transform(combined)

    reference_matrix = matrix[:len(references)]
    query_matrix = matrix[len(references):]

    print("Menghitung cosine similarity...")

    similarity_matrix = cosine_similarity(
        query_matrix,
        reference_matrix,
        dense_output=False
    )

    results = []

    print("\nMencari 3 kandidat teratas setiap mismatch...")

    for query_position, mismatch_id in enumerate(unmatched_ids):

        scores = similarity_matrix.getrow(query_position)

        if scores.nnz == 0:
            continue

        ranked = sorted(
            zip(scores.indices, scores.data),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        source = mismatch.loc[mismatch_id]

        for rank, (valid_position, score) in enumerate(ranked, start=1):
            book = valid.iloc[valid_position]

            results.append({
                "mismatch_id": mismatch_id,
                "rank": rank,
                "judul_mismatch": source["judul"],
                "author_mismatch": source["author"],
                "valid_id": book.get("id", ""),
                "judul_valid": book["judul"],
                "author_valid": book["author"],
                "scope": book.get("scope", ""),
                "bidang": book.get("bidang", ""),
                "title_similarity": round(float(score), 4)
            })

        if (query_position + 1) % 100 == 0:
            print(
                f"Progress: {query_position + 1}"
                f"/{len(unmatched_ids)}"
            )

    columns = [
        "mismatch_id",
        "rank",
        "judul_mismatch",
        "author_mismatch",
        "valid_id",
        "judul_valid",
        "author_valid",
        "scope",
        "bidang",
        "title_similarity"
    ]

    output = pd.DataFrame(results, columns=columns)

    output_path = BOOK_DIR / "audit_mismatch_fuzzy.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== HASIL FUZZY MATCHING ==========")
    print(f"Mismatch diperiksa       : {len(unmatched_ids)}")
    print(f"Kandidat ditemukan       : {len(output)}")
    print(
        "Mismatch dengan kandidat: "
        f"{output['mismatch_id'].nunique() if not output.empty else 0}"
    )

    if not output.empty:
        best = output[output["rank"] == 1]

        print("\nDistribusi skor kandidat terbaik:")
        print(
            pd.cut(
                best["title_similarity"],
                bins=[0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
                include_lowest=True
            ).value_counts().sort_index().to_string()
        )

    print(f"\nFile tersimpan: {output_path}")


if __name__ == "__main__":
    main()
