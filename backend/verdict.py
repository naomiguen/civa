"""
Keputusan akhir per referensi.

Urutan prioritas:
  1. Filter domain  : di luar Ilmu Komputer            -> DI LUAR SCOPE (tidak divonis valid/halu)
  2. Probabilitas ML (MURNI teks) digeser oleh bukti pangkalan data pada skala logit:
        verified  -> +evidence_weight   (judul ada & penulis/tahun/DOI konsisten)
        mismatch  -> -evidence_weight   (judul ada, tetapi penulis/tahun/DOI berbeda = pola manipulasi AI)
        not_found -> not_found_weight   (sudah dicari di semua sumber & tidak ada: indikasi lemah-sedang halusinasi;
                                         0 bila sumber eksternal tidak dapat dihubungi sama sekali)
     Dengan semua bobot 0, verdict sepenuhnya berasal dari model ML. Model ML sendiri tidak pernah melihat bukti ini.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from backend.config import settings
from backend.schemas import Evidence, Status

STATUS_LABEL = {"valid": "VALID", "halu": "HALU", "out_of_scope": "DI LUAR SCOPE", "unparsed": "TIDAK TERBACA"}
KIND_LABEL = {"journal": "Jurnal", "book": "Buku", "proceedings": "Prosiding", "preprint": "Preprint"}

OOS_REASON = "Referensi berada di luar batasan domain Ilmu Komputer (Out-of-Distribution)."


def evidence_shift(evidence: Evidence) -> float:
    """Pergeseran logit akibat bukti pangkalan data (positif = condong valid)."""
    if evidence.status == "verified":
        return settings.evidence_weight
    if evidence.status == "mismatch":
        return -settings.evidence_weight
    return 0.0 if evidence.unreachable else settings.not_found_weight


def fuse(p_valid: float, evidence: Evidence, threshold: float = 0.5) -> tuple[Status, float, float]:
    """(status, keyakinan pada status itu, p_valid akhir) dari probabilitas ML + bukti basis data."""
    p = min(max(float(p_valid), 1e-6), 1 - 1e-6)
    final = 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + evidence_shift(evidence))))
    status: Status = "valid" if final >= threshold else "halu"
    return status, (final if status == "valid" else 1.0 - final), final


def join_id(items: Sequence[str]) -> str:
    """['judul','penulis','tahun'] -> 'judul, penulis, dan tahun' (gaya Bahasa Indonesia)."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} dan {items[1]}"
    return ", ".join(items[:-1]) + f", dan {items[-1]}"


def build_reason(status: Status, evidence: Evidence, from_identifier: Optional[str] = None) -> str:
    """Teks 'Alasan' untuk kartu hasil (kalimatnya mengikuti desain UI).

    `from_identifier` ("doi" | "arxiv"): metadata diambil dari DOI/link yang ditempel, bukan dari teks referensi.
    """
    if status == "out_of_scope":
        return OOS_REASON
    if status == "unparsed":
        return "Alasan: Judul tidak dapat diekstraksi dari teks referensi ini."
    if from_identifier and status == "valid" and evidence.status == "verified":
        source = "arXiv" if from_identifier == "arxiv" else "DOI"
        return f"Alasan: {source} terdaftar; judul, penulis, dan tahun diambil dari metadata {source}."

    if evidence.status == "verified":
        matched = ["judul"] + [name for name, ok in (("penulis", evidence.author_match),
                                                     ("tahun", evidence.year_match),
                                                     ("DOI", evidence.doi_match)) if ok]
        if status == "valid":
            return f"Alasan: Referensi terverifikasi; {join_id(matched)} sesuai pangkalan data."
        return ("Alasan: Judul ditemukan di pangkalan data, tetapi pola teks menyerupai referensi "
                "halusinasi (klasifikasi ML).")

    if evidence.doi_issue == "missing":
        return ("Alasan: Indikasi manipulasi AI. DOI yang dicantumkan tidak terdaftar "
                "(tidak ditemukan di Crossref maupun registri DOI).")
    if evidence.doi_issue == "other_paper":
        other = f' "{evidence.record.title}"' if evidence.record else ""
        return f"Alasan: Indikasi manipulasi AI. DOI yang dicantumkan mengarah ke makalah lain:{other}."

    if evidence.status == "mismatch":
        fields = join_id(evidence.mismatch_fields)
        if status == "halu":
            return f"Alasan: Indikasi manipulasi AI. Judul ditemukan, namun {fields} tidak sesuai."
        return (f"Alasan: Judul ditemukan, namun {fields} berbeda dari pangkalan data; "
                "klasifikasi ML tetap menilai pola teks valid.")

    if evidence.unreachable:  # not_found karena semua API eksternal tidak dapat dihubungi
        verdict = "valid" if status == "valid" else "halusinasi"
        return (f"Alasan: Pangkalan data eksternal tidak dapat dihubungi; penilaian {verdict} hanya "
                "berdasarkan klasifikasi ML atas pola teks.")
    if status == "valid":  # not_found
        return ("Alasan: Judul belum ditemukan di pangkalan data, namun pola judul dan penulis "
                "konsisten dengan referensi asli (klasifikasi ML).")
    return ("Alasan: Judul tidak ditemukan di basis data lokal, Crossref, arXiv, OpenAlex, maupun DBLP; "
            "pola teks mirip referensi halusinasi (klasifikasi ML).")


def _initials(given: str) -> str:
    return " ".join(f"{token[0].upper()}." for token in given.replace(".", " ").replace("-", " ").split() if token[0].isalpha())


def authors_display(authors: Sequence[tuple[str, str]]) -> str:
    """[('Y','Zhang'),('L','Chen')] -> 'Zhang, Y. & Chen, L.' (gaya APA seperti pada desain)."""
    names = []
    for given, family in authors:
        initials = _initials(given)
        names.append(f"{family}, {initials}" if initials else family)
    if not names:
        return "Penulis tidak terdeteksi"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    if len(names) == 3:
        return f"{names[0]}, {names[1]}, & {names[2]}"
    return ", ".join(names[:3]) + ", dkk."
