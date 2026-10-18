"""Konfigurasi backend CIVA. Semua nilai bisa diganti lewat environment variable (tanpa edit kode)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Tahap 1: GROBID (Docker) -------------------------------------------------------
    grobid_url: str = os.getenv("CIVA_GROBID_URL", "http://localhost:8070").rstrip("/")
    grobid_timeout: float = _env_float("CIVA_GROBID_TIMEOUT", 90.0)  # detik; permintaan pertama bisa lambat

    # --- Tahap 2: Database lokal + API eksternal ----------------------------------------
    db_path: Path = Path(os.getenv("CIVA_DB_PATH", ROOT / "data" / "civa.db"))
    seed_dataset: Path = Path(os.getenv("CIVA_SEED_DATASET", ROOT / "dataset" / "dataset_asli_cs.xlsx"))
    external_timeout: float = _env_float("CIVA_EXTERNAL_TIMEOUT", 12.0)
    # Email kontak OPSIONAL untuk "polite pool" Crossref/OpenAlex (lebih cepat & stabil).
    # Tidak diisi otomatis; isi sendiri bila mau: set CIVA_CONTACT_EMAIL=email.anda@contoh.com
    contact_email: str = os.getenv("CIVA_CONTACT_EMAIL", "").strip()
    arxiv_min_interval: float = _env_float("CIVA_ARXIV_MIN_INTERVAL", 3.0)  # ketentuan arXiv: 1 request / 3 detik
    negative_cache_hours: float = _env_float("CIVA_NEGATIVE_CACHE_HOURS", 24.0)
    # Judul dianggap "ditemukan" (dan dapat diverifikasi) bila kemiripan teks >= nilai ini (0-1).
    title_match_threshold: float = _env_float("CIVA_TITLE_MATCH_THRESHOLD", 0.90)
    # Tuduhan "judul ada tetapi penulis/tahun berbeda" lebih berat: butuh judul hampir identik, karena makalah
    # lain dengan judul mirip ("Survey" vs "Review") bukan bukti manipulasi.
    mismatch_title_threshold: float = _env_float("CIVA_MISMATCH_TITLE_THRESHOLD", 0.97)
    # DOI sah tetapi milik makalah lain hanya dituduh bila judulnya JELAS berbeda (kemiripan di bawah nilai ini).
    doi_conflict_title_threshold: float = _env_float("CIVA_DOI_CONFLICT_TITLE_THRESHOLD", 0.60)

    # --- Keputusan akhir ----------------------------------------------------------------
    # Bukti pangkalan data menggeser logit probabilitas ML (model ML sendiri tetap murni teks).
    #   verified: +evidence_weight | mismatch: -evidence_weight | tidak ditemukan di mana pun: not_found_weight
    # Atur ketiganya ke 0 bila ingin verdict MURNI dari model ML.
    evidence_weight: float = _env_float("CIVA_EVIDENCE_WEIGHT", 4.0)
    not_found_weight: float = _env_float("CIVA_NOT_FOUND_WEIGHT", -1.5)
    valid_threshold: float = _env_float("CIVA_VALID_THRESHOLD", 0.5)

    # --- API ----------------------------------------------------------------------------
    max_references: int = _env_int("CIVA_MAX_REFERENCES", 10)
    concurrency: int = _env_int("CIVA_CONCURRENCY", 4)


settings = Settings()
