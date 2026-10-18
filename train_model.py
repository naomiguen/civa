#!/usr/bin/env python
"""
CIVA - Pelatihan model klasifikasi referensi (Valid / Halu)  --  MURNI TEKS.

ATURAN FITUR (sangat penting)
  * Input model HANYA teks kolom `judul` dan `author`.
  * judul  -> embedding MultiBERT 384-d      author -> embedding MultiBERT 384-d
    digabung (concatenate) = 768-d  ->  XGBoost / Random Forest / Logistic Regression.
  * TIDAK ADA fitur angka perbandingan, similarity, atau jarak ke database.
  * Target `status`: 1 = Valid (dataset_asli_cs.xlsx) ; 0 = Halu (dataset_hallucination_*.csv).

HASIL (folder models/)
  civa_pipeline.joblib        Pipeline sklearn [MultiBERTEmbedder -> classifier terkalibrasi]
  multibert_tokenizer.joblib  tokenizer MultiBERT
  domain_filter.joblib        filter domain (Semantic Thresholding) - terpisah dari classifier
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
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import civa_core
from civa_core import (CS_ANCHORS, EMB_DIM, FEATURE_DIM, MODEL_NAME, OTHER_ANCHORS, BalancedPlattClassifier,
                       DomainFilter, MultiBERTEmbedder, clean_authors, clean_title, to_initials)

ROOT = Path(__file__).resolve().parent
DOMAIN_KNN_PERCENTILE = 25  # ambang kNN = persentil ke-25 skor leave-one-out korpus Ilmu Komputer

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
def load_dataset(dataset_dir: Path, initials_rate: float, seed: int) -> pd.DataFrame:
    """Gabungkan dataset asli (status=1) dan halusinasi (status=0); bersihkan; augmentasi inisial."""
    real_path = dataset_dir / "dataset_asli_cs.xlsx"
    fake_candidates = sorted(dataset_dir.glob("dataset_hallucination*.csv"))
    if not real_path.exists() or not fake_candidates:
        sys.exit(f"Dataset tidak ditemukan di {dataset_dir} (butuh dataset_asli_cs.xlsx dan dataset_hallucination*.csv)")

    real = pd.read_excel(real_path)
    fake = pd.read_csv(fake_candidates[0])
    frame = pd.concat([
        pd.DataFrame({"judul": real["judul"], "author": real["author"].fillna(""), "status": 1}),
        pd.DataFrame({"judul": fake["judul"], "author": fake["author"].fillna(""), "status": 0}),
    ], ignore_index=True)
    raw_rows = len(frame)

    frame["judul"] = frame["judul"].astype(str).str.strip()
    frame["author"] = frame["author"].astype(str)
    key = frame["judul"].map(clean_title)
    frame = frame[(key.str.len() >= 4) & (frame["author"].map(clean_authors).str.len() > 0)]
    frame = frame.loc[~key.loc[frame.index].duplicated()].reset_index(drop=True)
    log(f"  baris mentah {raw_rows} -> setelah pembersihan {len(frame)} "
        f"(valid={int((frame.status == 1).sum())}, halu={int((frame.status == 0).sum())})")

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

    Ketidakseimbangan kelas (89% valid / 11% halu) ditangani dengan bobot kelas, bukan dengan membuang data.
    Grid sengaja berfokus pada regularisasi kuat karena sinyal teks lemah (model besar cepat overfit).
    """
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))])
    forest = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=seed)
    boost = XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                          min_child_weight=3, reg_lambda=5.0, scale_pos_weight=halu_over_valid,
                          tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed)
    if quick:
        grids = ({"clf__C": [0.01]}, {"min_samples_leaf": [3]}, {"max_depth": [3]})
    else:
        grids = ({"clf__C": [0.003, 0.01, 0.03, 0.1]},
                 {"min_samples_leaf": [1, 3, 5]},
                 {"max_depth": [3, 4], "n_estimators": [300, 500]})
    return {"LogisticRegression": (logreg, grids[0]), "RandomForest": (forest, grids[1]),
            "XGBoost": (boost, grids[2])}


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
    y = frame["status"].to_numpy()

    log("\n[2/7] Embedding MultiBERT")
    embedder = MultiBERTEmbedder(batch_size=64, show_progress=True)
    X = embed_dataset(embedder, frame, models_dir / "cache", use_cache=not args.no_cache)
    assert X.shape == (len(frame), FEATURE_DIM), X.shape
    log(f"  matriks fitur: {X.shape}  (judul 0:{EMB_DIM} | author {EMB_DIM}:{FEATURE_DIM})")

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, stratify=y,
                                           random_state=args.seed)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    log(f"  split stratified: train={len(train_idx)} (halu {int((y_train == 0).sum())}) | "
        f"test={len(test_idx)} (halu {int((y_test == 0).sum())})")

    # ---- 2. perbandingan model ----------------------------------------------------------------
    log(f"\n[3/7] Membandingkan Logistic Regression / Random Forest / XGBoost ({folds}-fold CV pada data train)")
    scoring = {"roc_auc": "roc_auc", "f1_macro": "f1_macro",
               "ap_halu": make_scorer(_ap_halu, response_method="predict_proba")}
    cv = StratifiedKFold(folds, shuffle=True, random_state=args.seed)
    halu_over_valid = float((y_train == 0).sum() / (y_train == 1).sum())
    rows, searches = [], {}
    for name, (estimator, grid) in candidates(args.seed, halu_over_valid, args.quick).items():
        t0 = time.time()
        search = GridSearchCV(estimator, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
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
    calibrated.fit(X_train, y_train)
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

    # ---- 4. ablasi modalitas -------------------------------------------------------------------
    log("\n[5/7] Ablasi: modalitas mana yang membawa sinyal? (algoritma & parameter terbaik, data uji)")
    ablation = {}
    for label, cols in (("judul_saja", slice(0, EMB_DIM)), ("author_saja", slice(EMB_DIM, FEATURE_DIM)),
                        ("judul+author", slice(0, FEATURE_DIM))):
        model = clone(best_estimator).fit(X_train[:, cols], y_train)
        ablation[label] = evaluate(y_test, model.predict_proba(X_test[:, cols])[:, 1])
        log(f"  {label:<13} AUC={ablation[label]['roc_auc']:.3f}  AP(halu)={ablation[label]['ap_halu']:.3f}")

    # ---- 5. filter domain ----------------------------------------------------------------------
    log("\n[6/7] Filter domain (Semantic Thresholding) - terpisah dari classifier")
    title_emb = X[:, :EMB_DIM]
    corpus = title_emb[y == 1]  # hanya referensi ASLI Ilmu Komputer
    anchors_cs = embedder.encode_texts([clean_title(a) for a in CS_ANCHORS])
    anchors_other = embedder.encode_texts([clean_title(a) for a in OTHER_ANCHORS])
    domain = DomainFilter(corpus, anchors_cs, anchors_other, knn_threshold=0.0, margin_threshold=0.0, k=5)
    knn_loo, margin_loo = domain.analyze(corpus, exclude_self=True)
    domain.knn_threshold = float(np.percentile(knn_loo, DOMAIN_KNN_PERCENTILE))
    false_oos = float((~domain.in_scope(knn_loo, margin_loo)).mean())
    sanity = {}
    for name, titles in (("non_cs", SANITY_NON_CS), ("cs", SANITY_CS)):
        k_, m_ = domain.analyze(embedder.encode_titles(titles))
        sanity[name] = float((~domain.in_scope(k_, m_)).mean())
    domain.meta = {"knn_percentile": DOMAIN_KNN_PERCENTILE, "corpus_size": int(len(corpus)),
                   "loo_false_out_of_scope": false_oos, "sanity_non_cs_flagged": sanity["non_cs"],
                   "sanity_cs_flagged": sanity["cs"]}
    log(f"  korpus CS: {len(corpus)} judul | ambang kNN={domain.knn_threshold:.3f} (persentil {DOMAIN_KNN_PERCENTILE} LOO)")
    log(f"  referensi CS asli yang salah diisolasi (leave-one-out): {false_oos:.1%}")
    log(f"  uji kewarasan tulisan tangan: non-CS terisolasi {sanity['non_cs']:.0%} | CS salah terisolasi {sanity['cs']:.0%}")

    # ---- 6. latih final & simpan ---------------------------------------------------------------
    log("\n[7/7] Melatih model final pada SELURUH data dan menyimpan artefak")
    final_clf = BalancedPlattClassifier(base=clone(best_estimator), cv=folds, random_state=args.seed).fit(X, y)
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
    assert loaded_domain.knn_threshold == domain.knn_threshold
    log("  verifikasi muat-ulang artefak: OK (prediksi identik)")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name, "best_params": searches[best_name].best_params_,
        "encoder": MODEL_NAME, "feature_dim": FEATURE_DIM, "features": "judul(384) + author(384), murni teks",
        "dataset": {"rows": int(len(frame)), "valid": int((y == 1).sum()), "halu": int((y == 0).sum()),
                    "initials_augmentation_rate": args.initials_rate, "test_size": args.test_size},
        "cv_folds": folds, "model_comparison": rows,
        "holdout_calibrated": holdout, "holdout_uncalibrated": uncalibrated, "ablation_holdout": ablation,
        "domain_filter": domain.meta | {"knn_threshold": domain.knn_threshold},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "seed": args.seed, "quick": args.quick,
        "catatan": [
            "Probabilitas dikalibrasi (Platt, bobot kelas seimbang): P(valid)=0.5 adalah batas keputusan prior seimbang.",
            "Dataset halusinasi bersifat sintetis; skor holdout mengukur kemampuan membedakan dataset ini, bukan "
            "halusinasi LLM di dunia nyata. Model tidak dapat memeriksa keberadaan sebuah makalah - itu tugas Tahap 2.",
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
