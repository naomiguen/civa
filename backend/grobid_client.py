"""
Tahap 1 - Ekstraksi metadata referensi dengan GROBID.

Endpoint: POST {GROBID}/api/processCitation  (form field `citations` = string referensi mentah)
Respons : fragmen TEI <biblStruct> (BUKAN dokumen TEI utuh). Catatan penting dari dokumentasi GROBID:
elemen <persName> membawa xmlns TEI sendiri sementara <biblStruct> tidak -> parser harus mengabaikan namespace.
Status  : 200 OK | 204 tidak ada yang bisa diekstrak | 503 semua thread sibuk (coba lagi setelah ~1 detik).
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from backend.config import settings
from backend.schemas import ParsedReference
from backend.textutil import normalize_doi, paren_year, parse_year, prepare_for_grobid, restore_title_case

log = logging.getLogger("civa.grobid")

_VENUE_PROCEEDINGS_RE = re.compile(
    r"proceed|conference|symposium|workshop|congress|konferensi|prosiding|seminar|\bproc\.", re.IGNORECASE
)


class GrobidError(RuntimeError):
    """GROBID menjawab, tetapi dengan error."""


class GrobidUnavailable(GrobidError):
    """GROBID tidak dapat dihubungi (Docker belum jalan / URL salah)."""


# --------------------------------------------------------------------------------------
# Parsing TEI (fungsi murni, mudah diuji)
# --------------------------------------------------------------------------------------
def _strip_namespaces(root: ET.Element) -> ET.Element:
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def _text(element: Optional[ET.Element]) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip() if element is not None else ""


def _first_title(parent: Optional[ET.Element]) -> tuple[str, str]:
    """(teks, level) judul pertama yang tidak kosong di bawah `parent`."""
    if parent is not None:
        for title in parent.findall("title"):
            if _text(title):
                return _text(title), title.get("level", "")
    return "", ""


def _parse_authors(parent: Optional[ET.Element]) -> list[tuple[str, str]]:
    authors: list[tuple[str, str]] = []
    if parent is None:
        return authors
    for author in parent.findall("author"):
        person = author.find("persName")
        if person is None:
            continue
        family = _text(person.find("surname"))
        given = " ".join(_text(f) for f in person.findall("forename") if _text(f))
        if family:
            authors.append((given, family))
    return authors


def _parse_year(bibl: ET.Element) -> Optional[int]:
    for date in bibl.iter("date"):
        year = parse_year(date.get("when")) or parse_year(_text(date))
        if year:
            return year
    return None


def parse_tei(xml_bytes: bytes | str, raw: str = "") -> Optional[ParsedReference]:
    """Ubah respons TEI GROBID menjadi ParsedReference. None bila XML tidak dapat dibaca."""
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")
    try:
        root = _strip_namespaces(ET.fromstring(xml_bytes))
    except ET.ParseError:
        log.warning("Respons GROBID bukan XML valid: %.120r", xml_bytes)
        return None
    bibl = root if root.tag == "biblStruct" else root.find(".//biblStruct")
    if bibl is None:
        return None

    analytic, monogr = bibl.find("analytic"), bibl.find("monogr")
    article_title, _ = _first_title(analytic)
    mono_title, mono_level = _first_title(monogr)

    if article_title:  # artikel jurnal / prosiding / bab buku
        title, venue = article_title, mono_title
        if mono_level == "j":
            kind = "journal"
        elif monogr is not None and (monogr.find("meeting") is not None or _VENUE_PROCEEDINGS_RE.search(venue)):
            kind = "proceedings"
        else:
            kind = "book" if mono_level == "m" else "journal"
        authors = _parse_authors(analytic) or _parse_authors(monogr)
    elif mono_title and mono_level != "j":  # monograf / buku (judul ada di <monogr>)
        title, venue = mono_title, ""
        kind = "proceedings" if _VENUE_PROCEEDINGS_RE.search(mono_title) else "book"
        authors = _parse_authors(monogr) or _parse_authors(analytic)
    else:  # hanya nama jurnal tanpa judul artikel -> tidak ada judul yang bisa diverifikasi
        title, venue, kind = "", mono_title, "other"
        authors = _parse_authors(analytic) or _parse_authors(monogr)

    doi = None
    for idno in bibl.iter("idno"):
        if idno.get("type", "").lower() == "doi":
            doi = normalize_doi(_text(idno))
            if doi:
                break
    return ParsedReference(raw=raw, title=title, authors=authors, year=_parse_year(bibl), doi=doi,
                           venue=venue, kind=kind)


# --------------------------------------------------------------------------------------
# Panggilan HTTP
# --------------------------------------------------------------------------------------
async def parse_citation(client: httpx.AsyncClient, raw: str) -> Optional[ParsedReference]:
    """Kirim satu referensi mentah ke GROBID. None = GROBID tidak menemukan apa pun (HTTP 204)."""
    url = f"{settings.grobid_url}/api/processCitation"
    timeout = httpx.Timeout(settings.grobid_timeout, connect=5.0)
    attempts = 3
    for attempt in range(attempts):
        try:
            response = await client.post(
                url, data={"citations": prepare_for_grobid(raw), "consolidateCitations": "0"},
                headers={"Accept": "application/xml"}, timeout=timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise GrobidUnavailable(
                f"GROBID tidak dapat dihubungi di {settings.grobid_url}. Pastikan container Docker GROBID berjalan."
            ) from exc
        except httpx.TimeoutException as exc:  # terhubung, tetapi tidak menjawab (mis. masih memuat model)
            if attempt < attempts - 1:
                log.warning("GROBID belum menjawab dalam %.0f dtk (percobaan %d/%d), mengulang...",
                            settings.grobid_timeout, attempt + 1, attempts)
                continue
            raise GrobidError(
                f"GROBID tidak menjawab dalam {settings.grobid_timeout:.0f} detik. Kemungkinan container masih memuat "
                "model atau kehabisan memori; tunggu 1-2 menit lalu coba lagi (periksa juga log container GROBID)."
            ) from exc
        except httpx.HTTPError as exc:
            raise GrobidError(f"Permintaan ke GROBID gagal: {type(exc).__name__}") from exc

        if response.status_code == 503 and attempt < attempts - 1:  # semua thread GROBID sibuk -> tunggu & ulangi
            await asyncio.sleep(1.0 + attempt)
            continue
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            raise GrobidError(f"GROBID mengembalikan HTTP {response.status_code}: {response.text[:200]}")
        parsed = parse_tei(response.content, raw)
        return apply_raw_overrides(parsed, raw) if parsed else None
    raise GrobidError("GROBID sedang sibuk (HTTP 503). Coba lagi sebentar.")


def apply_raw_overrides(parsed: ParsedReference, raw: str) -> ParsedReference:
    """DOI dan tahun yang tertulis jelas di teks pengguna lebih dapat dipercaya daripada tebakan model GROBID.

    Contoh: judul yang terpotong baris ('Periode 1995 / 2024') membuat GROBID mengambil 1995 sebagai tahun;
    atau tanda hubung Unicode pada DOI hasil salin PDF yang membuat DOI tampak 'berbeda'.
    """
    raw_doi = normalize_doi(raw)
    if raw_doi:
        parsed.doi = raw_doi
    raw_year = paren_year(raw)
    if raw_year:
        parsed.year = raw_year
    if parsed.title:  # kapitalisasi asli pengguna (teks ke GROBID mungkin sudah diubah dari HURUF KAPITAL)
        parsed.title = restore_title_case(parsed.title, raw)
    return parsed


async def warm_up(client: httpx.AsyncClient) -> None:
    """Pemanasan: satu referensi contoh ke GROBID agar permintaan pertama pengguna tidak kena inisialisasi lambat.

    Berjalan di latar belakang saat backend start; kegagalan diabaikan (GROBID bisa saja baru dinyalakan nanti).
    """
    try:
        started = asyncio.get_running_loop().time()
        await parse_citation(client, "Smith, J. (2020). A warm up citation for the parser. Journal of Tests, 1(1), 1-2.")
        log.info("GROBID siap (pemanasan %.1f dtk).", asyncio.get_running_loop().time() - started)
    except Exception as exc:  # noqa: BLE001
        log.info("Pemanasan GROBID dilewati: %s", exc)


async def is_alive(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get(f"{settings.grobid_url}/api/isalive", timeout=3.0)
        return response.status_code == 200 and response.text.strip().lower() == "true"
    except httpx.HTTPError:
        return False
