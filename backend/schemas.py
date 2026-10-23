"""Tipe data internal (dataclass) dan skema request/response API (pydantic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["valid", "halu", "out_of_scope", "unparsed"]
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
    # True bila sumber sendiri menyatakan ini literatur Ilmu Komputer (dataset seed, DBLP, arXiv cs.*, OpenAlex
    # field "Computer Science"). None = tidak diketahui; filter domain semantik yang memutuskan.
    cs_flag: Optional[bool] = None


@dataclass
class Evidence:
    """Keluaran Tahap 2: apa kata pangkalan data tentang referensi ini."""

    status: EvidenceStatus = "not_found"
    source: Optional[str] = None  # local | crossref | arxiv | openalex | dblp
    cached: bool = False  # True = diambil dari DB lokal (tanpa memanggil API)
    # True bila TIDAK ADA satu pun API eksternal yang berhasil dihubungi: "tidak ditemukan" lalu bukan bukti apa-apa.
    unreachable: bool = False
    # True bila salah satu rekaman yang cocok berasal dari sumber/kategori Ilmu Komputer (lihat Record.cs_flag).
    cs_hint: bool = False
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


class DomainOut(BaseModel):
    """Filter domain (Semantic Thresholding) - dihitung dari embedding judul, terpisah dari model ML."""

    p_cs: float  # peluang judul termasuk literatur Ilmu Komputer
    threshold: float  # di bawah ambang ini = di luar scope
    in_scope: bool
    forced_by_knowledge_base: bool = False  # True: ada di basis data Ilmu Komputer, otomatis in-scope


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
    label: str  # teks badge: "VALID" | "HALU" | "DI LUAR SCOPE" | "TIDAK TERBACA"
    # Referensi di luar scope tetap dinilai: "valid" | "halu" (None untuk status lain).
    scope_verdict: Optional[Literal["valid", "halu"]] = None
    # 0-1, keyakinan pada valid/halu (untuk out_of_scope: pada scope_verdict; None utk unparsed).
    confidence: Optional[float] = None
    # Perkiraan peluang halusinasi (0-1) untuk SEMUA referensi yang terbaca, termasuk "di luar scope" (None = tidak terbaca).
    halu_probability: Optional[float] = None
    warning: Optional[str] = None  # peringatan "perlu dicek ulang" (mis. di luar domain Ilmu Komputer)
    reason: str
    ml: Optional[MLOut] = None
    domain: Optional[DomainOut] = None
    evidence: Optional[EvidenceOut] = None


class Summary(BaseModel):
    total: int
    valid: int
    halu: int
    out_of_scope: int
    out_of_scope_valid: int = 0  # rincian "di luar scope" menurut scope_verdict
    out_of_scope_halu: int = 0
    unparsed: int
    valid_percent: float  # valid / total * 100
    # Persentase keseluruhan dari referensi yang terbaca: (valid + di luar scope yang valid) vs (halu + di luar scope
    # yang halu), sehingga hanya ada hijau dan merah. Jumlahnya 100 (0 dan 0 bila tidak ada yang terbaca).
    overall_valid_percent: float = 0.0
    overall_halu_percent: float = 0.0


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
