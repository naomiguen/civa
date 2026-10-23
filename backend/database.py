"""
Database lokal (SQLite) = Closed-Domain Knowledge Base CIVA.

* Terisi otomatis dari dataset/dataset_scope_6000.csv (referensi nyata Crossref/arXiv/OpenAlex: Ilmu Komputer dan
  bidang lain, kolom `scope`) saat pertama kali jalan.
* Setiap referensi yang ditemukan lewat API eksternal di-INSERT ke sini, sehingga pencarian berikutnya
  cukup dari database lokal (tanpa memanggil API lagi).
* Pencarian "tidak ditemukan" di-cache sementara (lookup_misses) agar API tidak dipanggil berulang untuk
  referensi yang sama (mis. saat tombol "Periksa Ulang Teks" ditekan).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from rapidfuzz import fuzz, process

from backend.schemas import Record
from backend.textutil import norm_key, normalize_doi
from civa_core import split_authors

log = logging.getLogger("civa.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_references (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    title_norm  TEXT NOT NULL,
    authors     TEXT NOT NULL DEFAULT '[]',        -- JSON: daftar nama lengkap
    year        INTEGER,
    doi         TEXT,
    url         TEXT,
    venue       TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'journal',   -- journal | book | proceedings | preprint
    source      TEXT NOT NULL,                     -- crossref | arxiv | openalex | dblp
    origin      TEXT NOT NULL DEFAULT 'cache',     -- seed (dataset) | cache (hasil API eksternal)
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    in_cs       INTEGER                            -- 1 = sumber menyatakan Ilmu Komputer ; NULL = tidak diketahui
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_kb_title_year ON kb_references(title_norm, COALESCE(year, 0));
CREATE INDEX IF NOT EXISTS ix_kb_doi ON kb_references(doi);
CREATE TABLE IF NOT EXISTS lookup_misses (
    title_norm  TEXT PRIMARY KEY,
    checked_at  REAL NOT NULL
);
"""

_INSERT_SQL = (
    "INSERT OR IGNORE INTO kb_references (title, title_norm, authors, year, doi, url, venue, kind, source, origin, in_cs) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index_titles: list[str] = []  # title_norm, indeks pencarian fuzzy di memori
        self._index_ids: list[int] = []
        with self._session() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            columns = {row["name"] for row in con.execute("PRAGMA table_info(kb_references)")}
            if "in_cs" not in columns:  # migrasi dari versi database sebelumnya
                con.execute("ALTER TABLE kb_references ADD COLUMN in_cs INTEGER")
            con.execute("UPDATE kb_references SET in_cs = 1 WHERE origin = 'seed' AND in_cs IS NULL")
        self._reload_index()

    # -- infrastruktur ---------------------------------------------------------------------
    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _reload_index(self) -> None:
        with self._session() as con:
            rows = con.execute("SELECT id, title_norm FROM kb_references").fetchall()
        with self._lock:
            self._index_ids = [r["id"] for r in rows]
            self._index_titles = [r["title_norm"] for r in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> Record:
        return Record(
            id=row["id"], title=row["title"], authors=json.loads(row["authors"] or "[]"), year=row["year"],
            doi=row["doi"], url=row["url"], venue=row["venue"], kind=row["kind"], source=row["source"],
            cs_flag=True if row["in_cs"] == 1 else None,
        )

    # -- baca ------------------------------------------------------------------------------
    def count(self) -> int:
        with self._session() as con:
            return int(con.execute("SELECT COUNT(*) FROM kb_references").fetchone()[0])

    def find(self, title: str, doi: Optional[str] = None, limit: int = 5) -> list[Record]:
        """Kandidat dari DB lokal: DOI persis -> judul persis -> judul mirip (fuzzy)."""
        key = norm_key(title)
        if not key:
            return []
        ids: list[int] = []
        with self._session() as con:
            if doi:
                ids += [r["id"] for r in con.execute("SELECT id FROM kb_references WHERE doi = ?", (doi,))]
            ids += [r["id"] for r in con.execute("SELECT id FROM kb_references WHERE title_norm = ?", (key,))]
        with self._lock:
            hits = process.extract(key, self._index_titles, scorer=fuzz.ratio, score_cutoff=80, limit=limit)
            ids += [self._index_ids[index] for _choice, _score, index in hits]
        ids = list(dict.fromkeys(ids))[: limit + 2]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._session() as con:
            rows = {r["id"]: r for r in con.execute(f"SELECT * FROM kb_references WHERE id IN ({marks})", ids)}
        return [self._to_record(rows[i]) for i in ids if i in rows]

    def recently_missed(self, title: str, ttl_hours: float) -> bool:
        key = norm_key(title)
        if not key or ttl_hours <= 0:
            return False
        with self._session() as con:
            row = con.execute("SELECT checked_at FROM lookup_misses WHERE title_norm = ?", (key,)).fetchone()
        return bool(row) and (time.time() - row["checked_at"]) < ttl_hours * 3600

    # -- tulis -----------------------------------------------------------------------------
    def insert(self, record: Record, origin: str = "cache") -> Optional[int]:
        """Simpan rekaman nyata dari API eksternal. Mengembalikan id baru, atau None bila sudah ada."""
        key = norm_key(record.title)
        if not key:
            return None
        row = (
            record.title.strip(), key, json.dumps(record.authors, ensure_ascii=False), record.year,
            normalize_doi(record.doi), record.url, record.venue or "", record.kind or "journal", record.source, origin,
            1 if record.cs_flag else None,
        )
        with self._lock, self._session() as con:
            cur = con.execute(_INSERT_SQL, row)
            new_id = cur.lastrowid if cur.rowcount else None
            con.execute("DELETE FROM lookup_misses WHERE title_norm = ?", (key,))
            if new_id:
                self._index_ids.append(new_id)
                self._index_titles.append(key)
        return new_id

    def remember_miss(self, title: str) -> None:
        key = norm_key(title)
        if key:
            with self._session() as con:
                con.execute("INSERT OR REPLACE INTO lookup_misses (title_norm, checked_at) VALUES (?, ?)",
                            (key, time.time()))

    # -- seed dari dataset -----------------------------------------------------------------
    def seed_if_needed(self, csv_path: Path) -> int:
        """Isi KB dari dataset seed (CSV) bila belum ada baris seed. Mengembalikan jumlah baris yang dimasukkan.

        Rekaman hasil API eksternal (origin='cache') tidak dihitung, jadi KB lama yang seed-nya dihapus ikut terisi lagi.
        """
        with self._session() as con:
            if con.execute("SELECT 1 FROM kb_references WHERE origin = 'seed' LIMIT 1").fetchone():
                return 0
        csv_path = Path(csv_path)
        if not csv_path.exists():
            log.warning("Dataset seed tidak ditemukan (%s); knowledge base dimulai tanpa seed.", csv_path)
            return 0
        import pandas as pd  # impor lazy: hanya perlu saat seeding

        def text(value: object) -> str:
            return "" if pd.isna(value) else str(value).strip()

        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        rows = []
        for item in frame.itertuples(index=False):
            title = text(item.judul)
            key = norm_key(title)
            if not key:
                continue
            link = text(item.link_referensi)  # DOI-URL (Crossref), URL arXiv, atau ISBN (buku)
            names = split_authors(text(item.author))
            rows.append((
                title, key, json.dumps(names, ensure_ascii=False),
                None if pd.isna(item.tahun) else int(item.tahun),
                normalize_doi(link), link if link.startswith("http") and "doi.org" not in link else None,
                "", "book" if text(item.jenis_referensi).lower() == "book" else "journal",
                text(item.sumber_api).lower(),  # "crossref" | "arxiv" | "openalex"
                "seed", 1 if text(item.scope) == "cs" else 0,  # in_cs: 1 = Ilmu Komputer, 0 = bidang lain
            ))
        with self._lock, self._session() as con:
            before = con.execute("SELECT COUNT(*) FROM kb_references").fetchone()[0]
            con.executemany(_INSERT_SQL, rows)
            inserted = con.execute("SELECT COUNT(*) FROM kb_references").fetchone()[0] - before
        self._reload_index()
        log.info("Knowledge base di-seed dari %s: %d referensi.", csv_path.name, inserted)
        return inserted
