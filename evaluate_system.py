#!/usr/bin/env python
"""
CIVA - Evaluasi SISTEM end-to-end (GROBID -> retrieval -> ML -> filter domain -> keputusan).

Berbeda dari train_model.py (yang hanya mengukur model ML), skrip ini mengukur sistem utuh pada sekumpulan referensi
yang kebenarannya diketahui:

  REAL (harus VALID)
    real_eksternal  referensi nyata CS (OpenAlex, 2019-2025, Inggris + Indonesia) yang TIDAK ada di dataset seed
    real_dataset    referensi nyata yang ada di dataset seed (jalur database lokal)
  PALSU (harus HALU)
    palsu_karangan  judul/penulis karangan penuh (ditulis tangan, diverifikasi tidak ada di Crossref)
    palsu_penulis   judul nyata + penulis dari makalah lain
    palsu_tahun     judul & penulis nyata + tahun digeser >= 3 tahun
    palsu_doi       judul, penulis, tahun nyata + DOI karangan

Set kedua (--set manipulasi): uji manipulasi dataset - referensi ASLI dari dataset_scope_6000.csv (CS & non-CS) dan
manipulasinya dari dataset_halu_6000_2kolom.csv (kata judul diacak / dihapus, susunan nama dibalik, penulis tidak
lengkap), ditambah referensi asli yang penulisnya dihapus (tanpa_penulis), diformat APA.

Cara kerja: skrip menyalakan backend sendiri (port & database SEMENTARA, jadi data/civa.db Anda tidak tersentuh) dan
memakai GROBID yang berjalan di CIVA_GROBID_URL (default http://localhost:8070). Butuh internet.

  python evaluate_system.py                    # susun set evaluasi (disimpan di reports/eval_set.json) lalu jalankan
  python evaluate_system.py --rebuild          # susun ulang set evaluasi
  python evaluate_system.py --set manipulasi   # uji manipulasi dataset (reports/manipulation_eval_set.json)
Keluaran: reports/system_evaluation.md, system_evaluation.json, system_eval_results.csv
          (--set manipulasi: reports/manipulation_evaluation.md, .json, manipulation_eval_results.csv)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from backend.config import settings  # noqa: E402  (juga membaca .env lokal, mis. kunci API OpenAlex)
from backend.textutil import norm_key, normalize_doi, title_similarity  # noqa: E402

REPORTS = ROOT / "reports"
SEED_CSV = ROOT / "dataset" / "dataset_scope_6000.csv"  # juga dataset seed basis data lokal
GROBID_URL = os.getenv("CIVA_GROBID_URL", "http://localhost:8070").rstrip("/")

# Referensi karangan penuh (judul sangat spesifik agar hampir pasti tidak ada). (judul, [penulis], tahun, jurnal)
FABRICATED = [
    ("Federated Wavelet Routing for Solar-Powered Mesh Networks under Intermittent Lunar Latency",
     ["Erik M. Karlsen", "Tobias R. Obuya", "Pierre Delacroix"], 2023, "Journal of Orbital Network Engineering"),
    ("A Topological Garbage Collector for Persistent Memory Graph Databases with Quantum-Resistant Snapshots",
     ["Lieke Verhoeven", "Suresh K. Anand"], 2022, "Transactions on Persistent Graph Systems"),
    ("Contrastive Pretraining of Sparse Transformers on Pollen-Count Time Series for Allergy Forecasting in Alpine Regions",
     ["Hannes Brandstetter", "David Nkemelu", "Yuki Ito"], 2024, "Alpine Journal of Applied Machine Learning"),
    ("Zero-Knowledge Proofs of Fairness for Gradient-Boosted Recommender Cascades in Municipal Library Systems",
     ["Chidi Okonkwo", "Astrid Lindqvist"], 2021, "Journal of Civic Recommender Systems"),
    ("Neuromorphic Compilation of Regular Expressions onto Memristive Crossbars for Edge Log Filtering",
     ["Nikos Papadakis", "Ming Zhao", "Rory Whitcombe"], 2025, "Memristive Computing Letters"),
    ("Self-Healing Microservice Meshes via Reinforcement Learning from Kubernetes Audit Trails of Greenhouse Controllers",
     ["Joao P. Ferreira", "Gurpreet Sandhu"], 2022, "Journal of Agricultural Cloud Operations"),
    ("Probabilistic Type Inference for Ancient Cuneiform Markup Languages Using Graph Attention",
     ["Thabo Mahlangu", "Katrin Voss", "Burak Aydin"], 2023, "Digital Philology and Computing Review"),
    ("Differentially Private Cache Eviction for Browser Extensions Operating on Encrypted Bookmark Graphs",
     ["Jonas Rasmussen", "Marta Ibarra"], 2020, "Privacy Engineering Quarterly"),
    ("Hyperbolic Embeddings of Pastry Recipe Dependency Graphs for Allergen-Aware Menu Generation",
     ["Sofia M. Delgado", "Ivan Kovalenko"], 2024, "Journal of Computational Gastronomy"),
    ("Latency-Aware Spiking Autoencoders for Acoustic Anomaly Detection in Tidal Turbine Gearboxes",
     ["Ola Hagen", "Peter Mwangi", "Linh Tran"], 2021, "Marine Energy Informatics"),
    ("Byzantine-Tolerant Consensus over Pigeon-Carried Storage Networks: A Formal Verification Study",
     ["Alberto Lombardi", "Kwame Osei"], 2022, "Journal of Avian Networking Protocols"),
    ("Curriculum Distillation of Large Language Models for Low-Resource Sign Language Gloss Segmentation",
     ["Elena Petrova", "Rahim Salim", "Haruto Nakamura"], 2025, "Sign Language Technology Review"),
    ("Optimasi Penjadwalan Armada Becak Listrik Menggunakan Algoritma Genetika Hibrida pada Kawasan Wisata Pesisir",
     ["Dimas P. Wicaksana", "Nurlela Siregar"], 2022, "Jurnal Teknologi Informasi dan Transportasi Pesisir"),
    ("Sistem Pakar Diagnosis Penyakit Tanaman Anggrek Berbasis Certainty Factor dengan Antarmuka Suara Bahasa Daerah",
     ["Rudi M. Hutapea", "Fitri A. Lestari", "Zaki Maulana"], 2023, "Jurnal Sistem Cerdas Hortikultura"),
    ("Implementasi Blockchain untuk Pencatatan Distribusi Garam Rakyat di Pulau Madura Berbasis Smart Contract Ringan",
     ["Ahmad Fauzan", "Dewi Prameswari"], 2024, "Jurnal Rekayasa Rantai Pasok Digital"),
    ("Klasifikasi Citra Batik Tulis Lasem Menggunakan EfficientNet dengan Augmentasi Motif Parang Sintetis",
     ["I Ketut Gunawan", "Endang Sulistyowati"], 2021, "Jurnal Visi Komputer Nusantara"),
    ("Rancang Bangun Aplikasi Pelaporan Sampah Pasar Tradisional Berbasis Geofencing dan Insentif Poin Digital",
     ["Rizky H. Nasution", "Wulan Kurniasih", "Samuel Pohan"], 2023, "Jurnal Aplikasi Mobile Lingkungan Perkotaan"),
    ("Analisis Sentimen Ulasan Wisata Kuliner Warung Tegal Menggunakan IndoBERT dengan Penanganan Slang Pesisir",
     ["Budi Ardiansyah", "Dian Rahmawati"], 2022, "Jurnal Pemrosesan Bahasa Alami Terapan"),
    ("Prediksi Debit Sungai Musiman Menggunakan LSTM Bidirectional dengan Fitur Fase Bulan dan Pasang Surut",
     ["Gilang A. Setiawan", "Maria T. Purba", "Luqman Hakim"], 2025, "Jurnal Hidroinformatika Cerdas"),
    ("Pengembangan Sistem Informasi Arsip Digital Naskah Lontar Berbasis OCR Aksara Bali dan Basis Data Graf",
     ["Ni Komang Suartika", "Putu Wirawan"], 2024, "Jurnal Humaniora Digital Nusantara"),
    ("Penerapan Algoritma Ant Colony pada Penentuan Rute Pengantaran Air Galon Isi Ulang di Perumahan Padat",
     ["Eko D. Simanjuntak", "Tuti Yuliana"], 2020, "Jurnal Optimasi Logistik Lokal"),
    ("Deteksi Kecurangan Transaksi Dompet Digital Koperasi Desa dengan Isolation Forest dan Aturan Fuzzy Musiman",
     ["Yoga W. Pratama", "Citra Anggraeni", "Mahyuddin Daulay"], 2023, "Jurnal Keamanan Keuangan Mikro"),
    ("Pemodelan Antrean Kapal Nelayan di Pelabuhan Perikanan Skala Kecil dengan Simulasi Diskrit dan Reinforcement Learning",
     ["Andi F. Lubis", "Hendra R. Santoso"], 2022, "Jurnal Simulasi Sistem Kelautan"),
    ("Sistem Rekomendasi Bibit Padi Gogo Berdasarkan Kandungan Hara Tanah Menggunakan Collaborative Filtering Berbobot",
     ["Jonathan S. Marbun", "Vina Amalia", "Bagas Kusuma"], 2021, "Jurnal Pertanian Presisi Digital"),
]

GROUP_LABEL = {"real_eksternal": "valid", "real_dataset": "valid", "palsu_karangan": "halu", "palsu_penulis": "halu",
               "palsu_tahun": "halu", "palsu_doi": "halu"}
GROUP_TITLE = {
    "real_eksternal": "Nyata, tidak ada di dataset (API eksternal)", "real_dataset": "Nyata, ada di dataset (lokal)",
    "palsu_karangan": "Palsu: karangan penuh", "palsu_penulis": "Palsu: judul nyata + penulis salah",
    "palsu_tahun": "Palsu: tahun digeser >= 3", "palsu_doi": "Palsu: DOI karangan",
}
# Set "manipulasi" (dataset_scope_6000 + dataset_halu_6000_2kolom); nama kelompok halu = jenis di train_model.py.
MANIPULATION_TITLE = {
    "asli_cs": "Asli, Ilmu Komputer", "asli_non_cs": "Asli, di luar Ilmu Komputer",
    "judul_diacak": "Halu: kata judul diacak", "judul_kata_dihapus": "Halu: kata judul dihapus",
    "nama_dibalik": "Halu: susunan nama penulis dibalik", "penulis_tidak_lengkap": "Halu: penulis tidak lengkap / satu nama",
    "tanpa_penulis": "Halu: tanpa nama penulis (disusun dari data asli)",
}
GROUP_TITLE.update(MANIPULATION_TITLE)
GROUP_LABEL.update({g: "valid" if g.startswith("asli") else "halu" for g in MANIPULATION_TITLE})


# --------------------------------------------------------------------------------------
# Penyusunan set evaluasi
# --------------------------------------------------------------------------------------
def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def fetch_openalex(client: httpx.Client, lang: str, n: int, seed: int) -> list[dict]:
    params = {"filter": f"primary_topic.field.id:17,language:{lang},type:article,has_doi:true,publication_year:2019-2025",
              "sample": n, "seed": seed, "per-page": min(n, 200),
              "select": "display_name,publication_year,authorships,doi,primary_location"}
    if settings.openalex_api_key:  # tanpa kunci: kuota harian bersama per IP
        params["api_key"] = settings.openalex_api_key
    for wait in (20, 60, 120, None):  # OpenAlex membatasi laju (429): tunggu lalu ulangi
        response = client.get("https://api.openalex.org/works", params=params)
        if response.status_code != 429 or wait is None:
            break
        print(f"  OpenAlex 429, menunggu {wait} detik ...", flush=True)
        time.sleep(wait)
    if response.status_code >= 400:  # tanpa URL di pesan: URL memuat kunci API
        sys.exit(f"OpenAlex menjawab {response.status_code}: {response.text[:300]}")
    out = []
    for work in response.json().get("results", []):
        title = _plain(work.get("display_name"))
        names = [(a.get("author") or {}).get("display_name", "") for a in work.get("authorships", [])]
        names = [n_ for n_ in names if n_][:4]  # 4 penulis PERTAMA apa adanya (nama tunggal ikut, seperti sitasi nyata)
        venue = _plain((((work.get("primary_location") or {}).get("source")) or {}).get("display_name", ""))
        doi = normalize_doi(work.get("doi"))
        if len(title) >= 25 and names and venue and doi and work.get("publication_year"):
            out.append({"title": title, "authors": names, "year": int(work["publication_year"]), "venue": venue,
                        "doi": doi, "lang": lang})
    return out


def _initials(tokens: list[str], spaced: bool) -> str:
    marks = [f"{t[0].upper()}." for t in tokens if t and t[0].isalpha()]
    return (" " if spaced else "").join(marks)


def authors_apa(names: list[str]) -> str:
    parts = [f"{n.split()[-1]}, {_initials(n.split()[:-1], True)}".rstrip(", ") for n in names]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", & " + parts[-1]


def authors_harvard_id(names: list[str]) -> str:
    parts = [f"{n.split()[-1]}, {_initials(n.split()[:-1], False)}".rstrip(", ") for n in names]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} dan {parts[1]}"
    return f"{parts[0]} dkk."


def format_reference(item: dict, rng: random.Random, doi: str | None, authors: list[str] | None = None,
                     year: int | None = None) -> str:
    names, year = authors or item["authors"], year or item["year"]
    if rng.random() < 0.6:  # APA
        text = f"{authors_apa(names)} ({year}). {item['title']}. {item['venue']}."
        return text + (f" https://doi.org/{doi}" if doi else "")
    text = f"{authors_harvard_id(names)} ({year}) “{item['title']},” {item['venue']}."  # Harvard (Indonesia)
    return text + (f" Tersedia pada: https://doi.org/{doi}." if doi else "")


def fake_doi(real_doi: str, rng: random.Random) -> str:
    return real_doi.split("/")[0] + "/" + "".join(rng.choice("abcdef0123456789") for _ in range(10))


def build_eval_set(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    seed_frame = pd.read_csv(SEED_CSV, encoding="utf-8-sig")
    seed_titles = {norm_key(t) for t in seed_frame["judul"].astype(str)}
    print("  mengambil sampel referensi nyata dari OpenAlex ...", flush=True)
    with httpx.Client(timeout=60, headers={"User-Agent": "CIVA-evaluation/1.0"}) as client:
        pool = fetch_openalex(client, "en", 130, seed) + fetch_openalex(client, "id", 130, seed + 1)
        pool = [p for p in pool if norm_key(p["title"]) not in seed_titles]
        rng.shuffle(pool)
        print(f"  kandidat nyata (di luar dataset): {len(pool)}", flush=True)

        # karangan penuh: pastikan benar-benar tidak ada di Crossref
        fabricated = []
        for title, authors, year, venue in FABRICATED:
            for attempt in range(4):  # Crossref kadang menjawab galat/HTML sesaat: ulangi dengan jeda
                response = client.get("https://api.crossref.org/works",
                                      params={"query.bibliographic": title, "rows": 3, "select": "title"})
                if response.status_code == 200 and response.headers.get("content-type", "").startswith("application/json"):
                    break
                time.sleep(5 * (attempt + 1))
            else:
                sys.exit(f"Crossref tidak menjawab dengan benar (HTTP {response.status_code}); coba lagi nanti.")
            best = max((title_similarity(title, (it.get("title") or [""])[0])
                        for it in response.json()["message"]["items"]), default=0.0)
            if best < 0.85:
                fabricated.append({"title": title, "authors": authors, "year": year, "venue": venue})
            else:
                print(f"   (dibuang, ternyata ada di Crossref: {title[:60]})")

    take = lambda n: [pool.pop() for _ in range(min(n, len(pool)))]  # noqa: E731
    real_ext, f_auth, f_year, f_doi = take(80), take(24), take(24), take(24)
    items: list[dict] = []

    def add(group: str, text: str, meta: dict) -> None:
        items.append({"group": group, "expected": GROUP_LABEL[group], "text": text, "meta": meta})

    for it in real_ext:
        add("real_eksternal", format_reference(it, rng, it["doi"] if rng.random() < 0.7 else None), it)
    for it in fabricated:
        doi = f"10.{rng.randint(10000, 99999)}/{rng.choice('abcdefgh')}{rng.randint(1000, 9999)}" if rng.random() < .5 else None
        add("palsu_karangan", format_reference(it, rng, doi), it)
    for it in f_auth:  # judul nyata, penulis dari makalah lain
        other = rng.choice(pool + real_ext)
        add("palsu_penulis", format_reference(it, rng, it["doi"] if rng.random() < .5 else None, authors=other["authors"]), it)
    for it in f_year:
        shift = rng.choice([-6, -5, -4, -3, 3, 4, 5])
        add("palsu_tahun", format_reference(it, rng, it["doi"] if rng.random() < .5 else None,
                                            year=min(2026, it["year"] + shift)), it)
    for it in f_doi:
        add("palsu_doi", format_reference(it, rng, fake_doi(it["doi"], rng)), it)

    kb = seed_frame[seed_frame["scope"] == "cs"].sample(20, random_state=seed)
    from civa_core import split_authors  # noqa: E402  (impor lambat: membawa sklearn)
    for row in kb.itertuples(index=False):
        names = split_authors("" if pd.isna(row.author) else row.author)[:4]  # 4 penulis pertama apa adanya
        if not names:
            continue
        link = "" if pd.isna(row.link_referensi) else str(row.link_referensi)
        item = {"title": str(row.judul).strip(), "authors": names, "year": int(row.tahun),
                "venue": "arXiv preprint" if str(row.sumber_api).lower() == "arxiv" else "Jurnal Ilmu Komputer"}
        add("real_dataset", format_reference(item, rng, normalize_doi(link) if rng.random() < .5 else None), item)
    rng.shuffle(items)
    return items


def build_manipulation_set(per_group: int = 20, seed: int = 7) -> list[dict]:
    """Referensi asli (dataset_scope) dan manipulasinya (dataset_halu), separuh CS separuh non-CS per kelompok, APA."""
    import train_model  # noqa: E402  (impor lambat: membawa sklearn/xgboost)
    rng = random.Random(seed)
    real = pd.read_csv(ROOT / "dataset" / train_model.VALID_FILE, encoding="utf-8-sig").fillna("")
    fake = pd.read_csv(ROOT / "dataset" / train_model.HALU_FILE, encoding="utf-8-sig").fillna("")
    source = train_model.link_sources(real["judul"].tolist(), real["author"].tolist(), fake["judul"].tolist(),
                                      fake["author"].tolist())
    pools = defaultdict(list)  # kelompok -> [(indeks sumber asli, judul, author)]
    for i, row in real.iterrows():
        if row["author"].strip():
            pools["asli_" + row["scope"]].append((i, row["judul"], row["author"]))
            pools["tanpa_penulis"].append((i, row["judul"], ""))
    for j, i in enumerate(source):
        if i >= 0:
            kind = train_model.perturbation_type(real["judul"][i], real["author"][i], fake["judul"][j], fake["author"][j])
            pools[kind].append((i, fake["judul"][j], fake["author"][j]))

    items: list[dict] = []
    for group in MANIPULATION_TITLE:
        picked = []
        for scope in ("cs", "non_cs"):
            pool = [p for p in pools[group] if real["scope"][p[0]] == scope]
            picked += rng.sample(pool, min(len(pool), per_group if group.startswith("asli") else per_group // 2))
        for i, title, author in picked:
            src = real.iloc[i]
            names = [n.strip() for n in author.split(",") if n.strip()]
            title, year = str(title).strip().rstrip("."), int(src["tahun"])
            venue = "arXiv preprint" if src["sumber_api"].lower() == "arxiv" else "Jurnal Ilmiah"
            text = f"{authors_apa(names)} ({year}). {title}. {venue}." if names else f"{title}. ({year}). {venue}."
            doi = normalize_doi(src["link_referensi"]) if rng.random() < 0.5 else None
            items.append({"group": group, "expected": GROUP_LABEL[group],
                          "text": text + (f" https://doi.org/{doi}" if doi else ""),
                          "meta": {"source_id": int(src["id"]), "scope": src["scope"], "bidang": src["bidang"]}})
    rng.shuffle(items)
    return items


# --------------------------------------------------------------------------------------
# Menjalankan backend terisolasi + evaluasi
# --------------------------------------------------------------------------------------
def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_backend(tmp: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = dict(os.environ, CIVA_DB_PATH=str(tmp / "eval.db"), PYTHONUNBUFFERED="1", CIVA_GROBID_URL=GROBID_URL)
    log = open(tmp / "backend.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    api = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            health = httpx.get(f"{api}/health", timeout=3).json()
            if health.get("model_loaded"):
                return proc, api
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    sys.exit(f"Backend evaluasi gagal start. Lihat {tmp / 'backend.log'}")


def evaluate(api: str, items: list[dict]) -> list[dict]:
    rows: list[dict] = []
    with httpx.Client(timeout=httpx.Timeout(900, connect=10)) as client:
        for start in range(0, len(items), 10):
            chunk = items[start:start + 10]
            text = "\n\n".join(it["text"] for it in chunk)
            for attempt in (1, 2):
                response = client.post(f"{api}/api/verify", json={"text": text})
                if response.status_code == 200:
                    break
                time.sleep(3)
            else:
                sys.exit(f"Backend mengembalikan {response.status_code}: {response.text[:300]}")
            results = response.json()["results"]
            if len(results) != len(chunk):
                sys.exit(f"Jumlah hasil ({len(results)}) != jumlah referensi ({len(chunk)}) pada batch {start}")
            for it, res in zip(chunk, results):
                rows.append({
                    "group": it["group"], "expected": it["expected"], "predicted": res["status"],
                    # di luar scope tetap dinilai valid/halu: itulah penilaian yang dibandingkan dengan kebenaran
                    "verdict": res.get("scope_verdict") or res["status"],
                    "confidence": res.get("confidence"), "ml_label": (res.get("ml") or {}).get("label"),
                    "ml_p_valid": (res.get("ml") or {}).get("probability_valid"),
                    "evidence": (res.get("evidence") or {}).get("status"),
                    "evidence_source": (res.get("evidence") or {}).get("source"),
                    "doi_issue": (res.get("evidence") or {}).get("doi_issue"),
                    "mismatch_fields": ", ".join((res.get("evidence") or {}).get("mismatch_fields") or []),
                    "in_scope": (res.get("domain") or {}).get("in_scope"),
                    "reason": res["reason"], "text": it["text"], "parsed_title": res.get("title"),
                })
            print(f"  {min(start + 10, len(items))}/{len(items)} referensi diperiksa", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    groups = {}
    for group, g in df.groupby("group"):
        expected = g["expected"].iloc[0]
        correct = (g["verdict"] == expected).mean()
        ml_correct = (g["ml_label"] == expected).mean()
        groups[group] = {
            "n": int(len(g)), "benar": float(correct), "ml_saja_benar": float(ml_correct),
            "status": {k: int(v) for k, v in g["predicted"].value_counts().items()},
            "bukti": {str(k): int(v) for k, v in g["evidence"].value_counts(dropna=False).items()},
            "pola": {k: int(v) for k, v in g["mismatch_fields"].str.split(", ").explode().value_counts().items() if k},
        }
    truth_fake, pred_fake = df["expected"] == "halu", df["verdict"] == "halu"
    tp, fp = int((truth_fake & pred_fake).sum()), int((~truth_fake & pred_fake).sum())
    fn, tn = int((truth_fake & ~pred_fake).sum()), int((~truth_fake & ~pred_fake).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ml_pred_fake = df["ml_label"] == "halu"
    ml_tp, ml_fp = int((truth_fake & ml_pred_fake).sum()), int((~truth_fake & ml_pred_fake).sum())
    real = df[~truth_fake]
    return {
        "total": int(len(df)), "groups": groups,
        "halu_detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                           "accuracy": (tp + tn) / len(df)},
        "ml_saja": {"recall": ml_tp / max(int(truth_fake.sum()), 1), "false_alarm": ml_fp / max(len(real), 1)},
        "real_false_alarm": float((real["verdict"] == "halu").mean()),
        "real_out_of_scope": float((real["predicted"] == "out_of_scope").mean()),
        "real_unparsed": float((real["predicted"] == "unparsed").mean()),
        "fake_unparsed": float((df[truth_fake]["predicted"] == "unparsed").mean()),
    }


SETS = {  # --set -> (berkas set evaluasi, awalan berkas laporan, judul laporan)
    "sistem": ("eval_set.json", "system", "Evaluasi sistem CIVA (end-to-end)"),
    "manipulasi": ("manipulation_eval_set.json", "manipulation", "Uji manipulasi dataset CIVA (end-to-end)"),
}
NOTES = {
    "sistem": [
        "- Referensi nyata diambil dari OpenAlex; sistem mencari ke Crossref/OpenAlex juga, sehingga ada unsur "
        "sirkular: angka 'nyata' mengukur kebenaran parsing + pencocokan + keputusan, bukan cakupan indeks.",
        "- Referensi palsu bersifat sintetis (karangan tangan dan korupsi terkontrol: penulis, tahun, DOI). "
        "Halusinasi LLM nyata bisa lebih beragam.",
        "- Referensi yang tidak berDOI atau tidak terindeks (buku, skripsi, jurnal lokal) kurang terwakili.",
    ],
    "manipulasi": [
        "- Referensi asli dari dataset_scope_6000.csv; referensi halu dari dataset_halu_6000_2kolom.csv, masing-masing "
        "ditautkan ke referensi asli sumbernya (tahun & DOI diambil dari sumber; 50% diberi DOI). Diformat APA.",
        "- 'Tanpa nama penulis' disusun dari referensi asli yang penulisnya dihapus, karena dataset halu praktis tidak "
        "memuat kasus ini.",
        "- Model ML final dilatih pada SELURUH dataset, termasuk baris-baris ini: kolom 'ML saja benar' optimistis.",
        "- Semua referensi halu di sini adalah manipulasi makalah NYATA, sehingga makalah aslinya bisa ditemukan di "
        "basis data lokal/API; yang diuji adalah pemeriksaan kecocokan Tahap 2 terhadap rekaman asli tersebut.",
    ],
}


def write_reports(rows: list[dict], summary: dict, seconds: float, which: str = "sistem") -> Path:
    set_file, prefix, title = SETS[which]
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / f"{prefix}_eval_results.csv", index=False, encoding="utf-8-sig")
    (REPORTS / f"{prefix}_evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    d, m = summary["halu_detection"], summary["ml_saja"]
    lines = [f"# {title}", "",
             f"{summary['total']} referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML, "
             f"filter domain) dalam {seconds / 60:.1f} menit.", "",
             "| Kelompok | n | Benar | ML saja benar | VALID | HALU | DI LUAR SCOPE | TIDAK TERBACA |", "|---|---|---|---|---|---|---|---|"]
    for group, info in summary["groups"].items():
        s = info["status"]
        lines.append(f"| {GROUP_TITLE[group]} | {info['n']} | {info['benar']:.0%} | {info['ml_saja_benar']:.0%} | "
                     f"{s.get('valid', 0)} | {s.get('halu', 0)} | {s.get('out_of_scope', 0)} | {s.get('unparsed', 0)} |")
    lines += ["", "## Deteksi halusinasi (positif = referensi palsu)", "",
              f"- Recall: **{d['recall']:.0%}** ({d['tp']} dari {d['tp'] + d['fn']} referensi palsu terdeteksi)",
              f"- Precision: **{d['precision']:.0%}** ({d['fp']} referensi nyata salah dituduh halu)",
              f"- F1: **{d['f1']:.2f}** | akurasi keseluruhan: {d['accuracy']:.0%}",
              f"- Referensi nyata: salah dituduh HALU {summary['real_false_alarm']:.0%}, "
              f"dinyatakan DI LUAR SCOPE {summary['real_out_of_scope']:.0%}, tidak terbaca {summary['real_unparsed']:.0%}",
              "- Referensi DI LUAR SCOPE tetap dinilai valid/halu (dengan peringatan); kolom Benar dan angka deteksi "
              "memakai penilaian itu.",
              f"- Pembanding, model ML saja (tanpa bukti pangkalan data): recall {m['recall']:.0%}, "
              f"alarm palsu {m['false_alarm']:.0%}", "",
              "## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok", ""]
    for group, info in summary["groups"].items():
        found = ", ".join(f"{k} ({v})" for k, v in info["pola"].items()) or "-"
        lines.append(f"- {GROUP_TITLE[group]}: {found}")
    lines += ["", "## Catatan metodologi", "", *NOTES[which],
              f"- GROBID: {GROBID_URL}. Set evaluasi tersimpan di reports/{set_file} (reproduksibel)."]
    path = REPORTS / f"{prefix}_evaluation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluasi end-to-end sistem CIVA.")
    parser.add_argument("--rebuild", action="store_true", help="susun ulang set evaluasi")
    parser.add_argument("--api", help="pakai backend yang sudah berjalan (default: nyalakan backend terisolasi)")
    parser.add_argument("--set", choices=list(SETS), default="sistem", dest="which",
                        help="sistem: set evaluasi umum | manipulasi: uji manipulasi dataset_scope/dataset_halu")
    parser.add_argument("--per-group", type=int, default=20, help="--set manipulasi: referensi per kelompok")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    set_path = REPORTS / SETS[args.which][0]
    if set_path.exists() and not args.rebuild:
        items = json.loads(set_path.read_text(encoding="utf-8"))
        print(f"Memuat set evaluasi: {len(items)} referensi ({set_path.name})")
    else:
        print("Menyusun set evaluasi ...")
        items = build_eval_set() if args.which == "sistem" else build_manipulation_set(args.per_group)
        set_path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        print("  komposisi:", dict(Counter(it["group"] for it in items)))

    try:
        alive = httpx.get(f"{GROBID_URL}/api/isalive", timeout=5).text.strip() == "true"
    except httpx.HTTPError:
        alive = False
    if not alive:
        sys.exit(f"GROBID tidak aktif di {GROBID_URL}. Jalankan container GROBID dulu.")

    started = time.time()
    proc, tmp = None, None
    try:
        if args.api:
            api = args.api.rstrip("/")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="civa_eval_"))
            print("Menyalakan backend terisolasi (database sementara) ...")
            proc, api = run_backend(tmp)
        print("Mengevaluasi ...")
        rows = evaluate(api, items)
    finally:
        if proc:
            proc.terminate()
    summary = summarize(rows)
    path = write_reports(rows, summary, time.time() - started, args.which)
    print("\n" + path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
