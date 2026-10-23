#!/usr/bin/env python
"""
CIVA - Pelatihan model klasifikasi referensi (Valid / Halu)  --  MURNI TEKS.

ATURAN FITUR (sangat penting)
  * Input model HANYA teks kolom `judul` dan `author`.
  * judul  -> embedding MultiBERT 384-d      author -> embedding MultiBERT 384-d
    digabung (concatenate) = 768-d  ->  XGBoost / Random Forest / Logistic Regression.
  * TIDAK ADA fitur angka perbandingan, similarity, atau jarak ke database.
  * Target `status`: 1 = Valid (dataset_scope_6000.csv) ; 0 = Halu (dataset_halu_6000_2kolom.csv).

DATA
  dataset_scope_6000.csv        referensi ASLI: 3.000 Ilmu Komputer + 3.000 dari 23 bidang lain (kolom `scope`)
  dataset_halu_6000_2kolom.csv  referensi HALU turunan data asli di atas: kata judul diacak, kata judul dihapus,
                                susunan nama penulis dibalik, nama penulis tidak lengkap / tinggal satu
  Tiap baris halu ditautkan ke referensi asli sumbernya, sehingga (1) deteksi diukur per jenis manipulasi,
  (2) referensi asli + turunannya selalu di sisi split yang sama (tanpa kebocoran), dan (3) baris "halu" yang
  ternyata identik dengan aslinya dibuang (label kontradiktif).

HASIL (folder models/)
  civa_pipeline.joblib        Pipeline sklearn [MultiBERTEmbedder -> classifier terkalibrasi]
  multibert_tokenizer.joblib  tokenizer MultiBERT
  domain_filter.joblib        filter domain (Semantic Thresholding: Ilmu Komputer vs non-CS) - terpisah dari classifier
  training_report.json, model_comparison.csv

PEMAKAIAN
  python train_model.py            # lengkap (grid hiperparameter penuh)
  python train_model.py --quick    # grid kecil, jauh lebih cepat
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, make_scorer,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import civa_core
from civa_core import (EMB_DIM, FEATURE_DIM, MODEL_NAME, BalancedPlattClassifier, DomainFilter, MultiBERTEmbedder,
                       clean_authors, clean_title, to_initials)

ROOT = Path(__file__).resolve().parent
VALID_FILE = "dataset_scope_6000.csv"
HALU_FILE = "dataset_halu_6000_2kolom.csv"
# Ambang filter domain: paling banyak sekian persen judul Ilmu Komputer (out-of-fold) yang boleh salah diisolasi.
SCOPE_MAX_FALSE_OOS = 0.05

# Nama jenis manipulasi (kolom `jenis`) untuk laporan.
JENIS_LABEL = {
    "judul_diacak": "kata judul diacak", "judul_kata_dihapus": "kata judul dihapus",
    "judul_diubah": "judul diubah lain", "nama_dibalik": "susunan nama penulis dibalik",
    "penulis_tidak_lengkap": "nama penulis tidak lengkap / satu nama", "tanpa_penulis": "tanpa nama penulis",
    "penulis_diubah": "penulis diubah lain", "tidak_tertaut": "tidak tertaut ke data asli",
}

# Daftar kecil judul tulisan tangan HANYA untuk uji kewarasan filter domain (tidak dipakai melatih apa pun).
SANITY_NON_CS = [
    "Pengaruh Pemberian Ekstrak Daun Sirih terhadap Penyembuhan Luka Sayat pada Tikus Putih",
    "Hubungan Pola Asuh Orang Tua dengan Perkembangan Motorik Anak Usia Dini",
    "Tinjauan Yuridis terhadap Perlindungan Hukum Konsumen dalam Transaksi Jual Beli Tanah",
    "Uji Aktivitas Antibakteri Ekstrak Etanol Kulit Jeruk terhadap Staphylococcus aureus",
    "Pertumbuhan dan Hasil Tanaman Padi Sawah pada Berbagai Dosis Pupuk Organik",
    "Gambaran Tingkat Kecemasan Pasien Pra Operasi di Rumah Sakit Umum Daerah",
    "Analisis Pengaruh Inflasi dan Suku Bunga terhadap Harga Saham Perusahaan Perbankan",
    "Nilai-Nilai Moral dalam Novel Laskar Pelangi Karya Andrea Hirata",
    "Efficacy of Metformin versus Placebo in Patients with Type 2 Diabetes Mellitus: A Randomized Trial",
    "Climate Change Impacts on Coral Reef Biodiversity in the Indo-Pacific Region",
    "Synthesis and Characterization of Zinc Oxide Nanoparticles Using Green Chemistry Methods",
    "The Role of Mitochondrial Dysfunction in Parkinson's Disease Progression",
    "Cultural Memory and Postcolonial Identity in Contemporary African Literature",
    "Judicial Review and the Separation of Powers in Constitutional Democracies",
    "A Meta-Analysis of Cognitive Behavioral Therapy for Adolescent Depression",
    "Gravitational Wave Signatures of Binary Neutron Star Mergers",
]
SANITY_CS = [
    "Deep Residual Learning for Image Recognition",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Penerapan Algoritma K-Means untuk Segmentasi Pelanggan pada Toko Online",
    "Sistem Informasi Manajemen Perpustakaan Berbasis Web Menggunakan Laravel",
    "Analisis Sentimen Twitter terhadap Kebijakan Pemerintah Menggunakan Naive Bayes",
    "Federated Learning with Differential Privacy for Mobile Health Applications",
    "A Survey of Blockchain-Based Access Control for Internet of Things",
    "Rancang Bangun Sistem Monitoring Suhu Berbasis Arduino dan Android",
    "Graph Neural Networks for Traffic Forecasting",
    "Optimizing Query Performance in Distributed Database Systems",
    "Implementasi Algoritma Rijndael pada Aplikasi Enkripsi Pesan Teks",
]


def log(message: str = "") -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------------------
def _tokens(text: object) -> list[str]:
    """Kata-kata huruf kecil, untuk membandingkan referensi halu dengan sumber aslinya."""
    return re.findall(r"\w+", clean_title(text))


def link_sources(source_titles: Sequence[str], source_authors: Sequence[str], titles: Sequence[str],
                 authors: Sequence[str]) -> np.ndarray:
    """Indeks referensi asli yang menjadi sumber tiap referensi halu (-1 bila tidak tertaut).

    Urutan: judul sama -> kata-katanya sama tetapi diacak -> kata judul halu (min. 3 kata unik) >= 80% berasal dari
    satu judul asli -> penulis sama persis dengan tepat satu referensi asli yang memuat semua kata judulnya (judul
    yang tinggal 1-2 kata setelah penghapusan).
    """
    keys = [tuple(_tokens(t)) for t in source_titles]
    exact, bags, postings, by_author = {}, {}, defaultdict(set), defaultdict(list)
    for i, (key, author) in enumerate(zip(keys, source_authors)):
        exact.setdefault(key, i)
        bags.setdefault(tuple(sorted(key)), i)
        by_author[tuple(_tokens(author))].append(i)
        for word in set(key):
            postings[word].add(i)
    out = np.full(len(titles), -1)
    for j, (title, author) in enumerate(zip(titles, authors)):
        key = tuple(_tokens(title))
        match = exact.get(key, bags.get(tuple(sorted(key))))
        if match is None and len(set(key)) >= 3:
            votes = Counter(i for word in set(key) for i in postings.get(word, ()))
            # seri dipecah ke indeks terkecil: hasil sama di setiap proses (urutan set str ikut hash acak Python)
            best, shared = max(votes.items(), key=lambda item: (item[1], -item[0])) if votes else (-1, 0)
            match = best if shared >= 0.8 * len(set(key)) else None
        if match is None and key:
            same_author = [i for i in by_author.get(tuple(_tokens(author)), ()) if not Counter(key) - Counter(keys[i])]
            match = same_author[0] if len(same_author) == 1 else None
        out[j] = -1 if match is None else match
    return out


def perturbation_type(source_title: object, source_author: object, title: object, author: object) -> str:
    """Jenis manipulasi referensi halu dibanding sumber aslinya (kunci JENIS_LABEL; "identik" = tidak diubah)."""
    src_t, t = _tokens(source_title), _tokens(title)
    if src_t != t:
        if sorted(src_t) == sorted(t):
            return "judul_diacak"
        return "judul_kata_dihapus" if not Counter(t) - Counter(src_t) else "judul_diubah"
    src_a, a = _tokens(source_author), _tokens(author)
    if not a:
        return "identik" if not src_a else "tanpa_penulis"
    if src_a == a:
        return "identik"
    if sorted(src_a) == sorted(a):
        return "nama_dibalik"
    return "penulis_tidak_lengkap" if not Counter(a) - Counter(src_a) else "penulis_diubah"


def load_dataset(dataset_dir: Path, initials_rate: float, seed: int) -> pd.DataFrame:
    """Data asli (status=1) + halu (status=0) -> kolom judul, author, status, scope, jenis, group."""
    valid_path, halu_path = dataset_dir / VALID_FILE, dataset_dir / HALU_FILE
    if not valid_path.exists() or not halu_path.exists():
        sys.exit(f"Dataset tidak ditemukan di {dataset_dir} (butuh {VALID_FILE} dan {HALU_FILE})")
    real = pd.read_csv(valid_path, encoding="utf-8-sig").fillna({"judul": "", "author": ""})
    fake = pd.read_csv(halu_path, encoding="utf-8-sig").fillna({"judul": "", "author": ""})

    source = link_sources(real["judul"].tolist(), real["author"].tolist(), fake["judul"].tolist(),
                          fake["author"].tolist())
    linked = source >= 0
    kinds = [perturbation_type(real["judul"].iat[i], real["author"].iat[i], t, a) if i >= 0 else "tidak_tertaut"
             for i, t, a in zip(source, fake["judul"], fake["author"])]
    frame = pd.concat([
        pd.DataFrame({"judul": real["judul"], "author": real["author"], "status": 1, "scope": real["scope"],
                      "jenis": "asli", "group": np.arange(len(real))}),
        pd.DataFrame({"judul": fake["judul"], "author": fake["author"], "status": 0,
                      "scope": np.where(linked, real["scope"].to_numpy()[source], None), "jenis": kinds,
                      "group": np.where(linked, source, len(real) + np.arange(len(fake)))}),
    ], ignore_index=True)
    log(f"  asli {len(real)} (scope: {real['scope'].value_counts().to_dict()}) | halu {len(fake)}, "
        f"{int(linked.sum())} tertaut ke referensi asli sumbernya")
    log("  jenis manipulasi pada data halu: " + ", ".join(f"{k}={v}" for k, v in Counter(kinds).most_common()))

    frame["judul"] = frame["judul"].astype(str).str.strip()
    frame["author"] = frame["author"].astype(str)
    title_key, author_key = frame["judul"].map(clean_title), frame["author"].map(clean_authors)
    pair = title_key + "\x1f" + author_key
    usable = (title_key.str.len() >= 4) & (author_key.str.len() > 0)
    # Baris "halu" yang sama dengan referensi asli = label kontradiktif: tidak bisa dipelajari, jadi dibuang.
    is_halu = frame["status"] == 0
    contradictory = is_halu & ((frame["jenis"] == "identik") | pair.isin(set(pair[~is_halu])))
    keep = usable & ~contradictory
    duplicate = keep & pair.where(keep).duplicated()
    keep &= ~duplicate
    log(f"  dibuang: {int(contradictory.sum())} halu identik dengan aslinya | {int((~usable & ~contradictory).sum())} "
        f"tanpa judul/penulis | {int(duplicate.sum())} duplikat")
    frame = frame[keep].reset_index(drop=True)
    log(f"  setelah pembersihan {len(frame)} baris (valid={int((frame.status == 1).sum())}, "
        f"halu={int((frame.status == 0).sum())})")

    # Augmentasi: sebagian author diubah ke gaya inisial ("Y. Zhang") -- sama rata untuk kedua kelas, jadi bukan
    # jalan pintas. Alasannya: input pengguna bergaya APA ("Zhang, Y.") jauh lebih sering berinisial daripada dataset.
    mask = np.random.default_rng(seed).random(len(frame)) < initials_rate
    frame.loc[mask, "author"] = frame.loc[mask, "author"].map(to_initials)
    log(f"  augmentasi inisial pada {int(mask.sum())} baris ({initials_rate:.0%}, acak, kedua kelas sama rata)")
    return frame


def embed_dataset(embedder: MultiBERTEmbedder, frame: pd.DataFrame, cache_dir: Path, use_cache: bool) -> np.ndarray:
    """Fitur 768-d = [embedding judul 384 | embedding author 384]. Hasil di-cache agar training ulang cepat."""
    digest = hashlib.sha1(json.dumps([MODEL_NAME, frame["judul"].tolist(), frame["author"].tolist()],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"features_{digest}.npy"
    if use_cache and cache_file.exists():
        log(f"  fitur diambil dari cache ({cache_file.name})")
        return np.load(cache_file)
    log("  menghitung embedding MultiBERT (judul lalu author) ...")
    titles = embedder.encode_titles(frame["judul"].tolist())
    authors = embedder.encode_authors(frame["author"].tolist())
    features = np.hstack([titles, authors]).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, features)
    return features


def grouped_holdout(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split stratified per GRUP: referensi asli dan semua turunan halunya selalu di sisi yang sama."""
    splitter = StratifiedGroupKFold(max(2, round(1 / test_size)), shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y, groups))


# --------------------------------------------------------------------------------------
# 2. Evaluasi
# --------------------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, p_valid: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p_valid >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, p_valid),
        "ap_halu": average_precision_score(1 - y_true, 1 - p_valid),  # PR-AUC kelas Halu (baseline = proporsi halu)
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "precision_halu": prec[0], "recall_halu": rec[0], "f1_halu": f1[0],
        "precision_valid": prec[1], "recall_valid": rec[1], "f1_valid": f1[1],
        "f1_macro": float(f1.mean()),
        "confusion": {"halu_dideteksi_halu": int(tn), "halu_lolos_sebagai_valid": int(fp),
                      "valid_salah_dituduh_halu": int(fn), "valid_dikenali_valid": int(tp)},
        "mean_p_valid_pada_valid": float(p_valid[y_true == 1].mean()),
        "mean_p_valid_pada_halu": float(p_valid[y_true == 0].mean()),
    }


def evaluate_by_kind(test: pd.DataFrame, p_valid: np.ndarray, threshold: float = 0.5) -> dict:
    """{"jenis|scope": {"n", "rate"}}: rate = % halu yang terdeteksi; untuk "asli" = % yang salah dituduh halu.

    scope "semua" = gabungan Ilmu Komputer dan di luar Ilmu Komputer.
    """
    flagged = p_valid < threshold
    scopes = test["scope"].fillna("?").to_numpy()
    out = {}
    for jenis, idx in test.groupby("jenis").indices.items():
        for scope in ("semua", "cs", "non_cs"):
            part = idx if scope == "semua" else idx[scopes[idx] == scope]
            if len(part):
                out[f"{jenis}|{scope}"] = {"n": int(len(part)), "rate": float(flagged[part].mean())}
    return out


def _ap_halu(y_true, p_valid):
    return average_precision_score(1 - np.asarray(y_true), 1 - np.asarray(p_valid))


def brief(m: dict) -> str:
    return (f"AUC={m['roc_auc']:.3f}  AP(halu)={m['ap_halu']:.3f}  F1(halu)={m['f1_halu']:.3f}  "
            f"F1-macro={m['f1_macro']:.3f}  bal.acc={m['balanced_accuracy']:.3f}  MCC={m['mcc']:.3f}")


# --------------------------------------------------------------------------------------
# 3. Kandidat model
# --------------------------------------------------------------------------------------
def candidates(seed: int, halu_over_valid: float, quick: bool) -> dict[str, tuple[object, dict]]:
    """Tiga algoritma yang dibandingkan (sesuai desain): Logistic Regression, Random Forest, XGBoost.

    Kelas hampir seimbang (data halu adalah turunan data asli); bobot kelas tetap dipakai untuk sisa selisihnya.
    """
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))])
    forest = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=seed)
    boost = XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          min_child_weight=3, reg_lambda=5.0, scale_pos_weight=halu_over_valid,
                          tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed)
    if quick:
        grids = ({"clf__C": [0.03]}, {"min_samples_leaf": [3]}, {"max_depth": [4]})
    else:
        grids = ({"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3]},
                 {"min_samples_leaf": [1, 3, 5]},
                 {"max_depth": [3, 5], "n_estimators": [300, 600]})
    return {"LogisticRegression": (logreg, grids[0]), "RandomForest": (forest, grids[1]),
            "XGBoost": (boost, grids[2])}


def scope_model(C: float = 0.1) -> Pipeline:
    """Classifier filter domain: Ilmu Komputer (1) vs disiplin lain (0) dari embedding judul 384-d."""
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model CIVA (murni teks: judul 384-d + author 384-d).")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--quick", action="store_true", help="grid hiperparameter kecil + 3-fold CV (cepat)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--initials-rate", type=float, default=0.35)
    parser.add_argument("--no-cache", action="store_true", help="hitung ulang embedding walau cache ada")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore", category=UserWarning)
    started = time.time()
    folds = 3 if args.quick else 5
    models_dir = civa_core.MODELS_DIR
    models_dir.mkdir(exist_ok=True)

    log("=" * 78)
    log("CIVA - training (murni teks: judul 384-d + author 384-d = 768-d)")
    log("=" * 78)

    # ---- 1. data & fitur ---------------------------------------------------------------------
    log("\n[1/7] Memuat dataset")
    frame = load_dataset(args.dataset_dir, args.initials_rate, args.seed)
    y, groups = frame["status"].to_numpy(), frame["group"].to_numpy()

    log("\n[2/7] Embedding MultiBERT")
    embedder = MultiBERTEmbedder(batch_size=64, show_progress=True)
    X = embed_dataset(embedder, frame, models_dir / "cache", use_cache=not args.no_cache)
    assert X.shape == (len(frame), FEATURE_DIM), X.shape
    log(f"  matriks fitur: {X.shape}  (judul 0:{EMB_DIM} | author {EMB_DIM}:{FEATURE_DIM})")

    train_idx, test_idx = grouped_holdout(y, groups, args.test_size, args.seed)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    g_train = groups[train_idx]
    log(f"  split stratified per grup (asli + turunannya tidak terpisah): train={len(train_idx)} "
        f"(halu {int((y_train == 0).sum())}) | test={len(test_idx)} (halu {int((y_test == 0).sum())})")

    # ---- 2. perbandingan model ----------------------------------------------------------------
    log(f"\n[3/7] Membandingkan Logistic Regression / Random Forest / XGBoost ({folds}-fold CV pada data train)")
    scoring = {"roc_auc": "roc_auc", "f1_macro": "f1_macro",
               "ap_halu": make_scorer(_ap_halu, response_method="predict_proba")}
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=args.seed)
    halu_over_valid = float((y_train == 0).sum() / (y_train == 1).sum())
    rows, searches = [], {}
    for name, (estimator, grid) in candidates(args.seed, halu_over_valid, args.quick).items():
        t0 = time.time()
        search = GridSearchCV(estimator, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train, groups=g_train)
        searches[name] = search
        best = search.best_index_
        test_metrics = evaluate(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
        rows.append({
            "model": name, "best_params": json.dumps(search.best_params_),
            "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
            "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][best],
            "cv_f1_macro": search.cv_results_["mean_test_f1_macro"][best],
            "cv_ap_halu": search.cv_results_["mean_test_ap_halu"][best],
            "test_roc_auc": test_metrics["roc_auc"], "test_ap_halu": test_metrics["ap_halu"],
            "test_f1_halu": test_metrics["f1_halu"], "test_f1_macro": test_metrics["f1_macro"],
            "fit_seconds": round(time.time() - t0, 1),
        })
        log(f"  {name:<19} CV AUC={rows[-1]['cv_roc_auc']:.3f}±{rows[-1]['cv_roc_auc_std']:.3f}  "
            f"CV F1-macro={rows[-1]['cv_f1_macro']:.3f}  | test AUC={test_metrics['roc_auc']:.3f}  "
            f"{search.best_params_}  ({rows[-1]['fit_seconds']}s)")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(models_dir / "model_comparison.csv", index=False)

    best_name = comparison.sort_values("cv_roc_auc", ascending=False).iloc[0]["model"]  # dipilih dari CV, bukan test
    best_estimator = searches[best_name].best_estimator_
    log(f"\n  >> model terbaik (berdasarkan CV ROC-AUC): {best_name}  {searches[best_name].best_params_}")

    # ---- 3. evaluasi holdout + kalibrasi -------------------------------------------------------
    log("\n[4/7] Evaluasi pada data uji (holdout) dengan probabilitas terkalibrasi")
    calibrated = BalancedPlattClassifier(base=clone(best_estimator), cv=folds, random_state=args.seed)
    calibrated.fit(X_train, y_train, groups=g_train)
    p_test = calibrated.predict_proba(X_test)[:, 1]
    holdout = evaluate(y_test, p_test)
    uncalibrated = evaluate(y_test, best_estimator.predict_proba(X_test)[:, 1])
    log(f"  terkalibrasi : {brief(holdout)}")
    log(f"  tanpa kalibrasi: {brief(uncalibrated)}")
    c = holdout["confusion"]
    log(f"  confusion (ambang 0.5): halu terdeteksi {c['halu_dideteksi_halu']} | halu lolos {c['halu_lolos_sebagai_valid']} | "
        f"valid dituduh halu {c['valid_salah_dituduh_halu']} | valid dikenali {c['valid_dikenali_valid']}")
    log(f"  rata-rata P(valid): pada referensi valid = {holdout['mean_p_valid_pada_valid']:.2f} | "
        f"pada referensi halu = {holdout['mean_p_valid_pada_halu']:.2f}  (makin jauh = makin terpisah)")

    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    by_kind = evaluate_by_kind(test_frame, p_test)
    by_scope = {}
    for scope in ("cs", "non_cs"):
        idx = np.flatnonzero(test_frame["scope"].to_numpy() == scope)
        by_scope[scope] = evaluate(y_test[idx], p_test[idx])
    log("  per jenis (data uji, ambang 0.5)                 n   semua |    CS | non-CS")
    for jenis in ["asli"] + [k for k in JENIS_LABEL if f"{k}|semua" in by_kind]:
        cells = [f"{100 * by_kind[f'{jenis}|{s}']['rate']:4.0f}%" if f"{jenis}|{s}" in by_kind else "    -"
                 for s in ("semua", "cs", "non_cs")]
        label = "ASLI yang salah dituduh halu" if jenis == "asli" else "terdeteksi: " + JENIS_LABEL[jenis]
        log(f"    {label:<45} {by_kind[f'{jenis}|semua']['n']:>4}  {cells[0]} | {cells[1]} | {cells[2]}")
    log(f"  AUC per scope: Ilmu Komputer {by_scope['cs']['roc_auc']:.3f} | "
        f"di luar Ilmu Komputer {by_scope['non_cs']['roc_auc']:.3f}")

    # ---- 4. ablasi modalitas -------------------------------------------------------------------
    log("\n[5/7] Ablasi: modalitas mana yang membawa sinyal? (algoritma & parameter terbaik, data uji)")
    ablation = {}
    for label, cols in (("judul_saja", slice(0, EMB_DIM)), ("author_saja", slice(EMB_DIM, FEATURE_DIM)),
                        ("judul+author", slice(0, FEATURE_DIM))):
        model = clone(best_estimator).fit(X_train[:, cols], y_train)
        ablation[label] = evaluate(y_test, model.predict_proba(X_test[:, cols])[:, 1])
        log(f"  {label:<13} AUC={ablation[label]['roc_auc']:.3f}  AP(halu)={ablation[label]['ap_halu']:.3f}")

    # ---- 5. filter domain ----------------------------------------------------------------------
    log("\n[6/7] Filter domain (Semantic Thresholding): Ilmu Komputer vs disiplin lain, dari embedding judul")
    # Judul halu ikut dipakai dengan scope sumber aslinya: judul yang diacak/terpotong tetap harus dikenali domainnya.
    known = frame["scope"].notna().to_numpy()
    y_scope = (frame["scope"] == "cs").to_numpy().astype(int)
    titles = X[:, :EMB_DIM]
    s_train, s_test = train_idx[known[train_idx]], test_idx[known[test_idx]]
    scope_search = GridSearchCV(scope_model(), {"clf__C": [0.01, 0.1, 1.0]}, scoring="roc_auc", cv=cv, n_jobs=1)
    scope_search.fit(titles[s_train], y_scope[s_train], groups=groups[s_train])
    best_c = scope_search.best_params_["clf__C"]
    oof = cross_val_predict(scope_model(best_c), titles[s_train], y_scope[s_train], groups=groups[s_train], cv=cv,
                            method="predict_proba")[:, 1]
    threshold = float(min(0.5, np.quantile(oof[y_scope[s_train] == 1], SCOPE_MAX_FALSE_OOS)))
    p_cs_test = scope_model(best_c).fit(titles[s_train], y_scope[s_train]).predict_proba(titles[s_test])[:, 1]
    cs_test, real_test = y_scope[s_test] == 1, y[s_test] == 1
    scope_eval = {
        "roc_auc": float(roc_auc_score(y_scope[s_test], p_cs_test)),
        "cs_salah_diisolasi": float((p_cs_test[cs_test] < threshold).mean()),
        "non_cs_terisolasi": float((p_cs_test[~cs_test] < threshold).mean()),
        "non_cs_terisolasi_asli": float((p_cs_test[~cs_test & real_test] < threshold).mean()),
        "non_cs_terisolasi_halu": float((p_cs_test[~cs_test & ~real_test] < threshold).mean()),
    }
    domain = DomainFilter(scope_model(best_c).fit(titles[known], y_scope[known]), threshold)
    sanity = {name: float((~domain.in_scope(domain.p_cs(embedder.encode_titles(items)))).mean())
              for name, items in (("non_cs", SANITY_NON_CS), ("cs", SANITY_CS))}
    domain.meta = {"trained_on": int(known.sum()), "C": best_c, "max_false_out_of_scope": SCOPE_MAX_FALSE_OOS,
                   "holdout": scope_eval, "sanity_non_cs_flagged": sanity["non_cs"], "sanity_cs_flagged": sanity["cs"]}
    log(f"  data: {int(known.sum())} judul (CS {int(y_scope[known].sum())} | non-CS {int((1 - y_scope[known]).sum())}), "
        f"C={best_c} | ambang P(CS) = {threshold:.3f} (maks. {SCOPE_MAX_FALSE_OOS:.0%} CS salah diisolasi, out-of-fold)")
    log(f"  data uji: AUC={scope_eval['roc_auc']:.3f} | CS salah diisolasi {scope_eval['cs_salah_diisolasi']:.1%} | "
        f"non-CS terisolasi {scope_eval['non_cs_terisolasi']:.1%} (asli {scope_eval['non_cs_terisolasi_asli']:.1%}, "
        f"halu {scope_eval['non_cs_terisolasi_halu']:.1%})")
    log(f"  uji kewarasan tulisan tangan: non-CS terisolasi {sanity['non_cs']:.0%} | CS salah terisolasi {sanity['cs']:.0%}")

    # ---- 6. latih final & simpan ---------------------------------------------------------------
    log("\n[7/7] Melatih model final pada SELURUH data dan menyimpan artefak")
    final_clf = BalancedPlattClassifier(base=clone(best_estimator), cv=folds, random_state=args.seed).fit(X, y, groups)
    pipeline = Pipeline([("embed", MultiBERTEmbedder(batch_size=64)), ("clf", final_clf)])

    joblib.dump(embedder.get_encoder().tokenizer, civa_core.TOKENIZER_PATH, compress=3)
    joblib.dump(pipeline, civa_core.PIPELINE_PATH, compress=3)
    joblib.dump(domain, civa_core.DOMAIN_PATH, compress=3)

    # Verifikasi: muat ulang dari disk (jalur yang sama dengan backend) & bandingkan dengan model di memori.
    loaded_pipeline, loaded_domain = civa_core.load_artifacts()
    probe = np.random.default_rng(args.seed).choice(len(frame), size=12, replace=False)
    reloaded = loaded_pipeline.predict_proba(frame.iloc[probe][["judul", "author"]])[:, 1]
    in_memory = final_clf.predict_proba(X[probe])[:, 1]
    assert np.allclose(reloaded, in_memory, atol=1e-4), "Artefak yang dimuat ulang tidak identik dengan model training!"
    assert loaded_domain.threshold == domain.threshold
    log("  verifikasi muat-ulang artefak: OK (prediksi identik)")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name, "best_params": searches[best_name].best_params_,
        "encoder": MODEL_NAME, "feature_dim": FEATURE_DIM, "features": "judul(384) + author(384), murni teks",
        "dataset": {"files": [VALID_FILE, HALU_FILE], "rows": int(len(frame)), "valid": int((y == 1).sum()),
                    "halu": int((y == 0).sum()), "per_jenis": frame["jenis"].value_counts().to_dict(),
                    "per_scope": {f"{scope}|{'valid' if status else 'halu'}": int(n) for (scope, status), n
                                  in frame.groupby(["scope", "status"]).size().items()},
                    "initials_augmentation_rate": args.initials_rate, "test_size": args.test_size,
                    "split": "StratifiedGroupKFold (referensi asli + turunan halunya satu grup)"},
        "cv_folds": folds, "model_comparison": rows,
        "holdout_calibrated": holdout, "holdout_uncalibrated": uncalibrated, "holdout_per_jenis": by_kind,
        "holdout_per_scope": by_scope, "ablation_holdout": ablation,
        "domain_filter": domain.meta | {"threshold": domain.threshold},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "seed": args.seed, "quick": args.quick,
        "catatan": [
            "Probabilitas dikalibrasi (Platt, bobot kelas seimbang): P(valid)=0.5 adalah batas keputusan prior seimbang.",
            "Dataset halusinasi bersifat sintetis (manipulasi data asli); skor holdout mengukur kemampuan membedakan "
            "dataset ini, bukan halusinasi LLM di dunia nyata. Model tidak dapat memeriksa keberadaan sebuah makalah - "
            "itu tugas Tahap 2.",
            "holdout_per_jenis: rate = persentase halu yang terdeteksi; untuk 'asli' = persentase yang salah dituduh halu.",
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (models_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")

    log("\n" + "=" * 78)
    log(f"SELESAI dalam {report['elapsed_seconds']:.0f} detik. Model terbaik: {best_name}")
    log(f"  holdout: {brief(holdout)}")
    log(f"  artefak : {civa_core.PIPELINE_PATH.relative_to(ROOT)}, {civa_core.TOKENIZER_PATH.relative_to(ROOT)}, "
        f"{civa_core.DOMAIN_PATH.relative_to(ROOT)}")
    log("  laporan : models/training_report.json, models/model_comparison.csv")
    log("=" * 78)


if __name__ == "__main__":
    main()
