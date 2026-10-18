"""
CIVA - Inti ML bersama (dipakai oleh train_model.py DAN backend/main.py).

Semua langkah yang mengubah teks menjadi fitur ada di sini, sehingga proses training dan
inference identik (tidak ada train/serve skew):

  1. clean_title() / clean_authors()  -> normalisasi teks mentah
  2. MultiBERTEmbedder                -> judul (384-d) + author (384-d) = 768-d, MURNI dari teks
  3. DomainFilter                     -> "Semantic Thresholding" untuk deteksi referensi di luar
                                         disiplin Ilmu Komputer. Ini filter terpisah dan TIDAK
                                         pernah menjadi fitur untuk model klasifikasi.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight

# --------------------------------------------------------------------------------------
# Konstanta & lokasi artefak
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "civa_pipeline.joblib"
TOKENIZER_PATH = MODELS_DIR / "multibert_tokenizer.joblib"
DOMAIN_PATH = MODELS_DIR / "domain_filter.joblib"
REPORT_PATH = MODELS_DIR / "training_report.json"

# Multilingual (termasuk Bahasa Indonesia) MiniLM berbasis BERT, keluaran 384 dimensi.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM = 384
FEATURE_DIM = 2 * EMB_DIM  # judul (384) + author (384) = 768
MAX_AUTHORS = 10  # batasi panjang daftar author agar jumlah author tidak menjadi "jalan pintas" model

# --------------------------------------------------------------------------------------
# Normalisasi teks
# --------------------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_NAME_SPLIT_RE = re.compile(r"\s*(?:,|;|&|\band\b|\bdan\b)\s*", re.IGNORECASE)
_AFFILIATION_RE = re.compile(
    r"universit|fakultas|faculty|institut|politeknik|polytechnic|departemen|department|jurusan|"
    r"program studi|prodi|school of|college|laborator|kementerian|akademi|sekolah tinggi|"
    r"computer science|engineering|informatika|informatics|ilmu komputer|\bdept\b|@",
    re.IGNORECASE,
)
_TRAILING_MARK_RE = re.compile(r"(?<=[^\W\d_])\d+$")  # penanda afiliasi: "Eko Yulianto3" -> "Eko Yulianto"
_STRIP_TITLE_CHARS = " \t\r\n\"'“”‘’«».,;:-–—"


def _basic_clean(text: object) -> str:
    text = unicodedata.normalize("NFKC", "" if text is None else str(text))
    text = text.replace("�", " ")  # karakter rusak akibat salah encoding
    return _WS_RE.sub(" ", text).strip()


def clean_title(text: object, lowercase: bool = True) -> str:
    """Rapikan judul: spasi ganda, tanda baca di ujung, dan (opsional) huruf kecil.

    Huruf kecil dipakai agar judul ALL-CAPS dan Title Case terbaca sama oleh model.
    """
    text = _basic_clean(text).strip(_STRIP_TITLE_CHARS)
    return text.lower() if lowercase else text


def _canon_name(name: str) -> str:
    """'R.G. Krisnadhi' -> 'R. G. Krisnadhi' ; 'Y Zhang' -> 'Y. Zhang' (inisial seragam)."""
    name = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z])", r"\1. ", name)
    tokens = [t + "." if len(t) == 1 and t.isalpha() else t for t in name.split()]
    return " ".join(tokens)


def split_authors(text: object) -> list[str]:
    """Pecah string author menjadi daftar nama, buang afiliasi/noise & duplikat."""
    names: list[str] = []
    seen: set[str] = set()
    for part in _NAME_SPLIT_RE.split(_basic_clean(text)):
        part = re.sub(r"[*†‡§¶#]+", "", part)
        part = _TRAILING_MARK_RE.sub("", part).strip(" .-–—")
        if len(part) < 2:
            continue
        if _AFFILIATION_RE.search(part) or re.search(r"\d", part) or len(part.split()) > 6:
            continue  # kemungkinan afiliasi/alamat/email, bukan nama orang
        part = _canon_name(part)
        key = part.lower()
        if key not in seen:
            seen.add(key)
            names.append(part)
    return names


def clean_authors(text: object, lowercase: bool = True, max_authors: int = MAX_AUTHORS) -> str:
    """Daftar author -> 'Nama Satu, Nama Dua, ...' (maks. `max_authors` nama pertama)."""
    out = ", ".join(split_authors(text)[:max_authors])
    return out.lower() if lowercase else out


def to_initials(text: object) -> str:
    """'Mehdi Sari, Joao Arwidarasto' -> 'M. Sari, J. Arwidarasto' (dipakai untuk augmentasi training)."""
    out = []
    for name in split_authors(text):
        parts = name.split()
        if len(parts) < 2:
            out.append(name)
            continue
        *given, surname = parts
        initials = " ".join(f"{g[0].upper()}." for g in given if g[0].isalpha())
        out.append(f"{initials} {surname}".strip())
    return ", ".join(out)


def authors_from_parts(authors: Iterable[tuple[str, str]]) -> str:
    """[(given, family), ...] dari GROBID -> 'Given Family, Given Family' (format yang sama dgn training)."""
    return ", ".join(" ".join(p for p in (given.strip(), family.strip()) if p) for given, family in authors)


# --------------------------------------------------------------------------------------
# Embedding MultiBERT -> fitur 768-d (MURNI TEKS)
# --------------------------------------------------------------------------------------
def _restore_tokenizer(encoder) -> bool:
    """Pasang tokenizer dari multibert_tokenizer.joblib ke encoder (tokenisasi sama persis dgn training).

    sentence-transformers >= 6 menyimpan tokenizer di `module.processor`; versi <= 5 di `module.tokenizer`.
    Kegagalan di sini tidak fatal: tokenizer bawaan model di hub identik dengan yang disimpan.
    """
    if not TOKENIZER_PATH.exists():
        return False
    try:
        tokenizer = joblib.load(TOKENIZER_PATH)
        module = encoder[0]
        if hasattr(module, "processor"):
            module.processor = tokenizer
        else:
            module.tokenizer = tokenizer
        return True
    except Exception:  # noqa: BLE001 - fallback ke tokenizer bawaan
        return False


class MultiBERTEmbedder(BaseEstimator, TransformerMixin):
    """Langkah pipeline: DataFrame[judul, author] -> ndarray (n, 768).

    768 = [embedding(judul) 384-d | embedding(author) 384-d]. Tidak ada fitur lain
    (tanpa similarity, jarak, atau angka dari database).

    Bobot encoder tidak ikut ter-pickle (±470 MB); hanya konfigurasinya. Encoder dimuat
    lazy dari cache HuggingFace, dan tokenizer dipulihkan dari multibert_tokenizer.joblib
    bila file tersebut ada sehingga tokenisasi sama persis dengan saat training.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 64,
                 device: str | None = None, lowercase: bool = True, show_progress: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.lowercase = lowercase
        self.show_progress = show_progress

    # -- encoder (lazy) ------------------------------------------------------------------
    def get_encoder(self):
        encoder = self.__dict__.get("_encoder")
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            try:  # cache lokal dulu: cepat dan tetap jalan tanpa internet
                encoder = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:  # noqa: BLE001 - belum pernah diunduh: unduh dari HuggingFace Hub
                encoder = SentenceTransformer(self.model_name, device=self.device)
            _restore_tokenizer(encoder)
            self._encoder = encoder
        return encoder

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_encoder", None)
        return state

    # -- API sklearn ---------------------------------------------------------------------
    def fit(self, X=None, y=None):  # stateless: tidak ada yang dipelajari
        return self

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.get_encoder().encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=self.show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_titles(self, titles: Sequence[object]) -> np.ndarray:
        return self.encode_texts([clean_title(t, self.lowercase) for t in titles])

    def encode_authors(self, authors: Sequence[object]) -> np.ndarray:
        return self.encode_texts([clean_authors(a, self.lowercase) for a in authors])

    def transform(self, X) -> np.ndarray:
        titles = self.encode_titles(list(X["judul"]))
        authors = self.encode_authors(list(X["author"]))
        return np.hstack([titles, authors]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Classifier terkalibrasi
# --------------------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class BalancedPlattClassifier(BaseEstimator, ClassifierMixin):
    """Classifier + kalibrasi Platt (sigmoid) agar angka probabilitas di UI bermakna.

    Kalibrator dilatih pada prediksi out-of-fold dengan bobot sampel SEIMBANG, sehingga P(valid) = 0,5 adalah
    batas keputusan untuk prior seimbang (bukan prior dataset yang 89% valid) dan probabilitas yang terlalu
    percaya diri (mis. dari pohon yang overfit) dikembalikan ke skala yang jujur.
    """

    def __init__(self, base=None, cv: int = 5, random_state: int = 42):
        self.base = base
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        out_of_fold = cross_val_predict(clone(self.base), X, y, cv=folds, method="predict_proba")
        positive = int(np.flatnonzero(np.unique(y) == 1)[0])
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(
            _logit(out_of_fold[:, positive]), y, sample_weight=compute_sample_weight("balanced", y))
        self.base_ = clone(self.base).fit(X, y)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_.predict_proba(X)[:, list(self.classes_).index(1)]
        valid = self.calibrator_.predict_proba(_logit(raw))[:, 1]
        return np.column_stack([1.0 - valid, valid])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------------------
# Filter domain: Semantic Thresholding (terpisah dari model klasifikasi)
# --------------------------------------------------------------------------------------
# Deskripsi disiplin (EN + ID) sebagai "jangkar" semantik. Judul yang lebih dekat ke disiplin lain
# daripada ke Ilmu Komputer menjadi kandidat DI LUAR SCOPE.
CS_ANCHORS = [
    "computer science", "software engineering and programming", "machine learning and artificial intelligence",
    "computer networks and cybersecurity", "databases and information systems", "algorithms and data structures",
    "computer vision and natural language processing", "internet of things and embedded systems",
    "ilmu komputer", "rekayasa perangkat lunak", "kecerdasan buatan dan pembelajaran mesin",
    "jaringan komputer dan keamanan siber", "sistem informasi dan basis data", "aplikasi web dan mobile",
]
OTHER_ANCHORS = [
    "medicine and clinical research", "biology and ecology", "chemistry and materials science",
    "physics and astronomy", "law and politics", "economics and finance", "education and teaching",
    "psychology and behavior", "literature, language and arts", "agriculture and farming",
    "nursing and public health", "sociology and culture", "environmental science",
    "business management and marketing",
    "kedokteran dan penelitian klinis", "biologi dan ekologi", "kimia dan ilmu material", "fisika dan astronomi",
    "hukum dan politik", "ekonomi dan keuangan", "pendidikan dan pengajaran", "psikologi dan perilaku",
    "sastra, bahasa dan seni", "pertanian dan perkebunan", "keperawatan dan kesehatan masyarakat",
    "sosiologi dan budaya", "ilmu lingkungan", "manajemen bisnis dan pemasaran",
]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class DomainFilter:
    """Semantic Thresholding: dua sinyal semantik pada embedding JUDUL harus sepakat agar sebuah referensi
    dinyatakan DI LUAR SCOPE (konservatif, supaya referensi Ilmu Komputer tidak salah diisolasi):

      knn_score = rata-rata cosine similarity ke k judul terdekat pada korpus Ilmu Komputer (basis data seed)
      margin    = kemiripan maks. ke jangkar Ilmu Komputer  -  kemiripan maks. ke jangkar disiplin lain

      DI LUAR SCOPE  <=>  margin < margin_threshold   DAN   knn_score < knn_threshold

    Referensi di luar Ilmu Komputer dengan begitu tidak keliru divonis "halusinasi" hanya karena tidak ada
    di basis data Ilmu Komputer. Filter ini terpisah dari model klasifikasi dan tidak pernah menjadi fiturnya.
    """

    def __init__(self, corpus_embeddings: np.ndarray, cs_anchor_embeddings: np.ndarray,
                 other_anchor_embeddings: np.ndarray, knn_threshold: float, margin_threshold: float = 0.0,
                 k: int = 5, meta: dict | None = None):
        self.corpus = _l2_normalize(np.asarray(corpus_embeddings, dtype=np.float32))
        self.cs_anchors = _l2_normalize(np.asarray(cs_anchor_embeddings, dtype=np.float32))
        self.other_anchors = _l2_normalize(np.asarray(other_anchor_embeddings, dtype=np.float32))
        self.knn_threshold = float(knn_threshold)
        self.margin_threshold = float(margin_threshold)
        self.k = int(k)
        self.meta = meta or {}

    def analyze(self, title_embeddings: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(knn_score, margin) per judul. `exclude_self=True` untuk leave-one-out saat kalibrasi ambang."""
        queries = _l2_normalize(np.asarray(title_embeddings, dtype=np.float32))
        sims = queries @ self.corpus.T
        if exclude_self:
            sims[np.arange(len(queries)), np.arange(len(queries))] = -1.0
        k = min(self.k, sims.shape[1])
        knn = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
        margin = (queries @ self.cs_anchors.T).max(axis=1) - (queries @ self.other_anchors.T).max(axis=1)
        return knn, margin

    def in_scope(self, knn_score: np.ndarray, margin: np.ndarray) -> np.ndarray:
        out_of_scope = (np.asarray(margin) < self.margin_threshold) & (np.asarray(knn_score) < self.knn_threshold)
        return ~out_of_scope


def load_artifacts():
    """Muat (pipeline, domain_filter) hasil train_model.py. Raise FileNotFoundError bila belum dilatih."""
    missing = [p.name for p in (PIPELINE_PATH, DOMAIN_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artefak model belum ada ({', '.join(missing)}). Jalankan dulu:  python train_model.py"
        )
    return joblib.load(PIPELINE_PATH), joblib.load(DOMAIN_PATH)
