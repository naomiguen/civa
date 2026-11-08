"""Tipe data internal (dataclass) dan skema request/response API (pydantic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["valid", "halu", "unparsed"]
EvidenceStatus = Literal["verified", "mismatch", "not_found"]


# --------------------------------------------------------------------------------------
# Tipe internal pipeline
# --------------------------------------------------------------------------------------
@dataclass
class ParsedReference:
    """Keluaran Tahap 1 (GROBID) untuk satu referensi."""

    raw: str
    title: str = ""
    authors: list[tuple[str, str]] = field(default_factory=list)  # (nama depan, nama belakang)
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: str = ""
    kind: str = "other"  # journal | book | proceedings | other
    # "doi" | "arxiv" bila metadata diambil dari pengenal (DOI/link arXiv) karena teks tidak memuat judul.
    from_identifier: Optional[str] = None
    identifier_missing: bool = False  # DOI-nya sendiri tidak terdaftar


@dataclass
class Record:
    """Rekaman referensi nyata (basis data lokal atau hasil API eksternal)."""

    title: str
    authors: list[str]  # nama lengkap, mis. "Yue Zhang"
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: str = ""
    kind: str = "journal"  # journal | book | proceedings | preprint
    source: str = "seed"  # seed | crossref | arxiv | openalex | dblp
    id: Optional[int] = None


@dataclass
class Evidence:
    """Keluaran Tahap 2: apa kata pangkalan data tentang referensi ini."""

    status: EvidenceStatus = "not_found"
    source: Optional[str] = None  # local | crossref | arxiv | openalex | dblp
    cached: bool = False  # True = diambil dari DB lokal (tanpa memanggil API)
    # True bila TIDAK ADA satu pun API eksternal yang berhasil dihubungi: "tidak ditemukan" lalu bukan bukti apa-apa.
    unreachable: bool = False
    # Hasil pemeriksaan DOI langsung: "missing" = DOI tidak terdaftar; "other_paper" = DOI mengarah ke makalah lain.
    doi_issue: Optional[str] = None
    title_similarity: Optional[float] = None
    author_match: Optional[bool] = None  # None = tidak bisa dibandingkan
    year_match: Optional[bool] = None
    doi_match: Optional[bool] = None
    mismatch_fields: list[str] = field(default_factory=list)  # subset {"penulis", "tahun", "DOI"}
    record: Optional[Record] = None


# --------------------------------------------------------------------------------------
# Skema API
# --------------------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks daftar pustaka mentah (maks. 10 referensi).")


class RecordOut(BaseModel):
    title: str
    authors: list[str]
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: str = ""
    kind: str = ""
    source: str = ""


class EvidenceOut(BaseModel):
    status: EvidenceStatus
    source: Optional[str] = None
    cached: bool = False
    unreachable: bool = False
    doi_issue: Optional[str] = None
    title_similarity: Optional[float] = None
    author_match: Optional[bool] = None
    year_match: Optional[bool] = None
    doi_match: Optional[bool] = None
    mismatch_fields: list[str] = []
    record: Optional[RecordOut] = None


class MLOut(BaseModel):
    """Prediksi model ML (MURNI teks judul + author)."""

    label: Literal["valid", "halu"]
    probability_valid: float
    probability_halu: float


class ReferenceResult(BaseModel):
    index: int
    raw: str
    title: Optional[str] = None
    authors: list[str] = []
    authors_display: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    type: str = "Lainnya"  # Jurnal | Buku | Prosiding | Preprint | Lainnya
    status: Status
    label: str  # teks badge: "VALID" | "HALU" | "TIDAK TERBACA"
    confidence: Optional[float] = None  # 0-1, keyakinan pada label akhir (None utk unparsed)
    halu_probability: Optional[float] = None  # perkiraan peluang halusinasi 0-1 (None = tidak terbaca)
    reason: str
    ml: Optional[MLOut] = None
    evidence: Optional[EvidenceOut] = None


class Summary(BaseModel):
    total: int
    valid: int
    halu: int
    unparsed: int
    valid_percent: float  # valid / total * 100


class VerifyResponse(BaseModel):
    summary: Summary
    results: list[ReferenceResult]
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    grobid_available: bool
    grobid_url: str
    knowledge_base_records: int
    model_name: Optional[str] = None
    best_classifier: Optional[str] = None
