"""
Tahap 2 - Multi-Source Retrieval & Caching.

Urutan pencarian judul hasil ekstraksi GROBID:
    1. Database lokal (SQLite)                      -> bila terverifikasi, selesai (tanpa memanggil API)
    2. API eksternal, BERURUTAN: Crossref -> arXiv -> OpenAlex -> DBLP
       - berhenti di sumber pertama yang memverifikasi referensi
       - rekaman nyata yang judulnya cocok langsung di-INSERT ke database lokal (cache permanen)
       - kegagalan satu API (timeout/429/dll) tidak menghentikan pipeline: lanjut ke API berikutnya
    3. Tidak ditemukan di mana pun -> dicatat sementara (negative cache) agar tidak menghujani API.

Untuk menambah sumber baru cukup tulis `async def search_xxx(client, ref) -> list[Record]` lalu daftarkan di PROVIDERS.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Awaitable, Callable, Optional
from urllib.parse import quote

import httpx

from backend.config import settings
from backend.database import KnowledgeBase
from backend.schemas import Evidence, ParsedReference, Record
from backend.textutil import (author_overlap, extract_arxiv_id, norm_key, normalize_doi, parse_year,
                              title_similarity)

log = logging.getLogger("civa.retrieval")

_TAG_RE = re.compile(r"<[^>]+>")
_RANK = {"verified": 2, "mismatch": 1, "not_found": 0}


def _plain(text: object) -> str:
    """Buang markup HTML/MathML yang kadang ada di judul Crossref."""
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub("", "" if text is None else str(text)))).strip()


# --------------------------------------------------------------------------------------
# Pembandingan referensi <-> rekaman (menghasilkan bukti untuk 'Alasan' di UI)
# --------------------------------------------------------------------------------------
def compare(ref: ParsedReference, record: Record, source: str, cached: bool) -> Evidence:
    """Bandingkan referensi hasil ekstraksi dengan satu rekaman nyata.

    verified : judul cocok DAN penulis/tahun/DOI tidak bertentangan
    mismatch : judul cocok, tetapi penulis/tahun/DOI berbeda (pola khas sitasi hasil manipulasi AI)
    not_found: judul tidak cukup mirip
    """
    similarity = title_similarity(ref.title, record.title)
    if similarity < settings.title_match_threshold:
        return Evidence(status="not_found", title_similarity=similarity)

    overlap = author_overlap(ref.authors, record.authors)
    author_ok = None if overlap is None else overlap >= 0.5
    year_ok = None if ref.year is None or record.year is None else abs(ref.year - record.year) <= 1
    doi_ok = None
    if normalize_doi(ref.doi) and normalize_doi(record.doi):
        doi_ok = normalize_doi(ref.doi) == normalize_doi(record.doi)

    mismatch = [name for name, ok in (("penulis", author_ok), ("tahun", year_ok), ("DOI", doi_ok)) if ok is False]
    if mismatch and similarity < settings.mismatch_title_threshold:
        # Judul hanya mirip & metadata berbeda: kemungkinan besar makalah lain -> bukan bukti apa pun.
        return Evidence(status="not_found", title_similarity=similarity)
    return Evidence(
        status="mismatch" if mismatch else "verified", source=source, cached=cached, title_similarity=similarity,
        author_match=author_ok, year_match=year_ok, doi_match=doi_ok, mismatch_fields=mismatch, record=record,
        cs_hint=bool(record.cs_flag),
    )


def _score(evidence: Evidence) -> tuple[int, float, bool]:
    """Urutan prioritas deterministik: status, kemiripan judul, lalu sumber yang menyatakan Ilmu Komputer."""
    return _RANK[evidence.status], evidence.title_similarity or 0.0, bool(evidence.record and evidence.record.cs_flag)


def _forgive_doi(evidence: Evidence) -> Evidence:
    """DOI pengguna terbukti sah, jadi DOI rekaman yang berbeda (versi preprint/terbit) bukan bukti manipulasi."""
    if "DOI" in evidence.mismatch_fields:
        evidence.mismatch_fields = [f for f in evidence.mismatch_fields if f != "DOI"]
        evidence.doi_match = None
        if not evidence.mismatch_fields and evidence.status == "mismatch":
            evidence.status = "verified"
    return evidence


def best_of(candidates: list[Evidence]) -> Optional[Evidence]:
    usable = [c for c in candidates if c.status != "not_found"]
    return max(usable, key=_score, default=None)


# --------------------------------------------------------------------------------------
# HTTP util
# --------------------------------------------------------------------------------------
async def _get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """GET dengan satu kali retry untuk 429/5xx. Melempar exception bila tetap gagal."""
    for attempt in range(2):
        response = await client.get(url, params=params, timeout=settings.external_timeout)
        if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            retry_after = response.headers.get("Retry-After", "")
            await asyncio.sleep(min(float(retry_after), 5.0) if retry_after.isdigit() else 1.5)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("unreachable")


class _Throttle:
    """Jeda minimum antar-request (arXiv meminta maks. 1 request / 3 detik)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self.interval - (time.monotonic() - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


_arxiv_throttle = _Throttle(settings.arxiv_min_interval)

_CROSSREF_KIND = {
    "journal-article": "journal", "proceedings-article": "proceedings", "posted-content": "preprint",
    "book": "book", "monograph": "book", "edited-book": "book", "book-chapter": "book", "reference-book": "book",
}
_OPENALEX_KIND = {"article": "journal", "book": "book", "book-chapter": "book", "preprint": "preprint",
                  "dissertation": "book", "proceedings-article": "proceedings"}
_DBLP_KIND = {"Journal Articles": "journal", "Conference and Workshop Papers": "proceedings",
              "Books and Theses": "book", "Parts in Books or Collections": "book",
              "Informal and Other Publications": "preprint"}


# --------------------------------------------------------------------------------------
# Penyedia data eksternal
# --------------------------------------------------------------------------------------
async def search_crossref(client: httpx.AsyncClient, ref: ParsedReference) -> list[Record]:
    params = {"query.bibliographic": ref.title, "rows": 5,
              "select": "DOI,title,author,issued,container-title,type,URL"}
    if settings.contact_email:
        params["mailto"] = settings.contact_email
    items = (await _get(client, "https://api.crossref.org/works", params)).json().get("message", {}).get("items", [])
    return [rec for rec in (_crossref_record(item) for item in items) if rec]


def _crossref_record(item: dict) -> Optional[Record]:
    titles = item.get("title") or []
    if not titles:
        return None
    authors = [" ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
               for a in item.get("author", [])]
    date_parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
    return Record(
        title=_plain(titles[0]), authors=[a for a in authors if a], year=parse_year(date_parts[0]),
        doi=normalize_doi(item.get("DOI")), url=item.get("URL"),
        venue=_plain((item.get("container-title") or [""])[0]),
        kind=_CROSSREF_KIND.get(item.get("type", ""), "journal"), source="crossref",
    )


async def _fetch(client: httpx.AsyncClient, url: str, params: Optional[dict] = None) -> Optional[httpx.Response]:
    """GET tanpa melempar exception; diulang sekali untuk galat jaringan / 429 / 5xx. None bila tetap gagal."""
    for attempt in range(2):
        try:
            response = await client.get(url, params=params, timeout=settings.external_timeout)
        except httpx.HTTPError:
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            return None
        if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            await asyncio.sleep(1.5)
            continue
        return response
    return None


async def resolve_doi(client: httpx.AsyncClient, doi: str) -> tuple[str, Optional[Record]]:
    """Periksa DOI itu sendiri (bukan pencarian judul).

    ('found', Record)        DOI terdaftar di Crossref dan metadatanya didapat
    ('elsewhere', None)      DOI terdaftar di registri lain (DataCite, dll.) - sah, tanpa metadata
    ('missing', None)        DOI tidak terdaftar sama sekali (Crossref 404 DAN registri handle doi.org 'tidak ada')
    ('unknown', None)        tidak dapat dipastikan (jaringan/rate-limit) -> jangan dijadikan bukti
    """
    path = quote(doi, safe="/")
    params = {"mailto": settings.contact_email} if settings.contact_email else None
    response = await _fetch(client, f"https://api.crossref.org/works/{path}", params)
    if response is None:
        return "unknown", None
    if response.status_code == 200:
        try:
            return "found", _crossref_record(response.json().get("message", {}))
        except ValueError:
            return "unknown", None
    if response.status_code != 404:
        return "unknown", None
    handle = await _fetch(client, f"https://doi.org/api/handles/{path}")  # mungkin terdaftar di registri DOI lain
    if handle is None:
        return "unknown", None
    try:
        code = handle.json().get("responseCode") if handle.status_code == 200 else None
    except ValueError:
        code = None
    if code == 1:
        return "elsewhere", None
    if handle.status_code == 404 or code == 100:
        return "missing", None
    return "unknown", None


async def search_arxiv(client: httpx.AsyncClient, ref: ParsedReference) -> list[Record]:
    words = [w for w in norm_key(ref.title).split() if len(w) > 2][:10]
    if not words:
        return []
    await _arxiv_throttle.wait()
    response = await _get(client, "https://export.arxiv.org/api/query",
                          {"search_query": " AND ".join(f"ti:{w}" for w in words), "max_results": 5})
    return _parse_arxiv_atom(response.content)


def _parse_arxiv_atom(content: bytes) -> list[Record]:
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    records = []
    for entry in ET.fromstring(content).findall("a:entry", ns):
        title = _plain(entry.findtext("a:title", "", ns))
        if not title or title.lower() == "error":  # arXiv membalas entri "Error" untuk id yang tidak ada
            continue
        category = entry.find("x:primary_category", ns)
        term = category.get("term", "") if category is not None else ""
        records.append(Record(
            title=title, authors=[_plain(a.findtext("a:name", "", ns)) for a in entry.findall("a:author", ns)],
            year=parse_year(entry.findtext("a:published", "", ns)), doi=normalize_doi(entry.findtext("x:doi", "", ns)),
            url=entry.findtext("a:id", "", ns), venue="arXiv", kind="preprint", source="arxiv",
            cs_flag=True if term.startswith("cs.") else None,
        ))
    return records


# --------------------------------------------------------------------------------------
# Referensi yang HANYA berupa pengenal (DOI / link arXiv): ambil metadatanya dari pengenal itu
# --------------------------------------------------------------------------------------
_DATACITE_KIND = {"Preprint": "preprint", "JournalArticle": "journal", "Book": "book", "BookChapter": "book",
                  "ConferencePaper": "proceedings", "ConferenceProceeding": "proceedings"}


def _flip_name(name: str) -> str:
    """'Vaswani, Ashish' -> 'Ashish Vaswani'."""
    family, _, given = name.partition(",")
    return f"{given.strip()} {family.strip()}".strip() if given else name.strip()


async def resolve_datacite(client: httpx.AsyncClient, doi: str) -> Optional[Record]:
    """Metadata DOI yang terdaftar di DataCite (mis. 10.48550/arXiv.*, data, preprint)."""
    response = await _fetch(client, f"https://api.datacite.org/dois/{quote(doi, safe='/')}")
    try:
        attrs = response.json().get("data", {}).get("attributes", {}) if response and response.status_code == 200 else {}
    except ValueError:
        return None
    titles = attrs.get("titles") or []
    title = _plain(titles[0].get("title")) if titles else ""
    if not title:
        return None
    names = [" ".join(p for p in (c.get("givenName"), c.get("familyName")) if p) or _flip_name(c.get("name", ""))
             for c in attrs.get("creators", [])]
    container = attrs.get("container") or {}
    return Record(
        title=title, authors=[n for n in names if n], year=parse_year(attrs.get("publicationYear")), doi=doi,
        url=f"https://doi.org/{doi}", venue=_plain(container.get("title") or attrs.get("publisher") or ""),
        kind=_DATACITE_KIND.get((attrs.get("types") or {}).get("resourceTypeGeneral", ""), "journal"),
        source="datacite",
    )


def _parsed_from_record(raw: str, record: Record, doi: Optional[str], via: str) -> ParsedReference:
    authors = []
    for name in record.authors:
        tokens = name.split()
        if tokens:
            authors.append((" ".join(tokens[:-1]), tokens[-1]))
    return ParsedReference(raw=raw, title=record.title, authors=authors, year=record.year, doi=doi or record.doi,
                           venue=record.venue, kind=record.kind, from_identifier=via)


async def resolve_identifier(client: httpx.AsyncClient, raw: str) -> tuple[str, Optional[ParsedReference], Optional[Record]]:
    """Ubah DOI / link arXiv pada `raw` menjadi metadata referensi.

    Mengembalikan (status, ParsedReference | None, Record | None) dengan status:
      found    metadata didapat (Crossref, DataCite, atau arXiv)
      missing  DOI-nya tidak terdaftar sama sekali
      unknown  pengenal ada tetapi tidak dapat dipastikan (jaringan/rate-limit/registri tanpa metadata)
      none     tidak ada DOI maupun id arXiv pada teks
    """
    doi = normalize_doi(raw)
    arxiv_id = extract_arxiv_id(raw)
    if arxiv_id:  # arXiv dulu: respons arXiv membawa kategori (cs.*) yang tidak ada di DataCite/Crossref
        try:
            await _arxiv_throttle.wait()
            response = await _get(client, "https://export.arxiv.org/api/query", {"id_list": arxiv_id})
            records = _parse_arxiv_atom(response.content)
        except Exception as exc:  # noqa: BLE001
            log.warning("arXiv id %s gagal: %s", arxiv_id, exc)
            records = []
        if records:
            return "found", _parsed_from_record(raw, records[0], records[0].doi or doi, "arxiv"), records[0]
    if doi:
        state, record = await resolve_doi(client, doi)
        if state == "elsewhere":  # terdaftar di registri lain: coba DataCite
            record = await resolve_datacite(client, doi)
            state = "found" if record else "elsewhere"
        if state == "found" and record:
            return "found", _parsed_from_record(raw, record, doi, "doi"), record
        if state == "missing":
            return "missing", ParsedReference(raw=raw, doi=doi, from_identifier="doi", identifier_missing=True), None
    return ("unknown" if (doi or arxiv_id) else "none"), None, None


async def search_openalex(client: httpx.AsyncClient, ref: ParsedReference) -> list[Record]:
    params = {"search": ref.title, "per-page": 5,
              "select": "display_name,publication_year,authorships,doi,type,primary_location,primary_topic"}
    if settings.contact_email:
        params["mailto"] = settings.contact_email
    results = (await _get(client, "https://api.openalex.org/works", params)).json().get("results", [])
    records = []
    for work in results:
        title = _plain(work.get("display_name"))
        if not title:
            continue
        location = work.get("primary_location") or {}
        records.append(Record(
            title=title, authors=[(a.get("author") or {}).get("display_name", "") for a in work.get("authorships", [])],
            year=work.get("publication_year"), doi=normalize_doi(work.get("doi")),
            url=location.get("landing_page_url"), venue=_plain((location.get("source") or {}).get("display_name")),
            kind=_OPENALEX_KIND.get(work.get("type", ""), "journal"), source="openalex",
            cs_flag=True if (((work.get("primary_topic") or {}).get("field") or {}).get("display_name")
                             == "Computer Science") else None,
        ))
    return records


async def search_dblp(client: httpx.AsyncClient, ref: ParsedReference) -> list[Record]:
    words = norm_key(ref.title).split()[:12]
    if not words:
        return []
    data = (await _get(client, "https://dblp.org/search/publ/api",
                       {"q": " ".join(words), "format": "json", "h": 5})).json()
    hits = (data.get("result", {}).get("hits", {}) or {}).get("hit", []) or []
    records = []
    for hit in hits:
        info = hit.get("info", {})
        authors = (info.get("authors") or {}).get("author", [])
        authors = [authors] if isinstance(authors, dict) else authors
        records.append(Record(
            title=_plain(info.get("title", "")).rstrip("."), authors=[a.get("text", "") for a in authors],
            year=parse_year(info.get("year")), doi=normalize_doi(info.get("doi")), url=info.get("ee") or info.get("url"),
            venue=_plain(info.get("venue", "")) if isinstance(info.get("venue"), str) else "",
            kind=_DBLP_KIND.get(info.get("type", ""), "journal"), source="dblp",
            cs_flag=True,  # DBLP = bibliografi Ilmu Komputer
        ))
    return [r for r in records if r.title]


Provider = Callable[[httpx.AsyncClient, ParsedReference], Awaitable[list[Record]]]
PROVIDERS: list[tuple[str, Provider]] = [
    ("crossref", search_crossref),
    ("arxiv", search_arxiv),
    ("openalex", search_openalex),
    ("dblp", search_dblp),
]


# --------------------------------------------------------------------------------------
# Alur Tahap 2
# --------------------------------------------------------------------------------------
async def retrieve_evidence(client: httpx.AsyncClient, kb: KnowledgeBase, ref: ParsedReference) -> Evidence:
    """Database lokal dulu; bila belum terverifikasi, cari ke API eksternal secara berurutan."""
    doi = normalize_doi(ref.doi)

    # 2a. Database lokal
    local = [compare(ref, rec, "local", cached=True)
             for rec in await asyncio.to_thread(kb.find, ref.title, doi)]
    best = best_of(local)
    cs_hint = any(c.cs_hint for c in local)  # salah satu rekaman cocok berasal dari sumber Ilmu Komputer
    if best and best.status == "verified":
        best.cs_hint = cs_hint
        return best

    # 2a'. Periksa DOI itu sendiri (bukti terkuat bila pengguna mencantumkannya)
    doi_valid = False  # DOI terbukti sah -> perbedaan DOI dengan rekaman (mis. preprint vs terbit) bukan tuduhan
    if doi:
        state, doi_record = await resolve_doi(client, doi)
        if state == "found" and doi_record:
            on_doi = compare(ref, doi_record, "crossref", cached=False)
            if on_doi.status != "not_found":  # DOI mengarah ke makalah dengan judul yang sama
                await asyncio.to_thread(kb.insert, doi_record)
                on_doi.cs_hint = on_doi.cs_hint or cs_hint
                return on_doi
            if (on_doi.title_similarity or 0.0) < settings.doi_conflict_title_threshold:
                return Evidence(status="mismatch", source="crossref", title_similarity=on_doi.title_similarity,
                                doi_match=False, mismatch_fields=["DOI"], record=doi_record, doi_issue="other_paper",
                                cs_hint=cs_hint)
            doi_valid = True  # judul mirip tetapi belum meyakinkan: lanjut pencarian judul
        elif state == "missing":
            return Evidence(status="mismatch", source="crossref", doi_match=False, mismatch_fields=["DOI"],
                            doi_issue="missing", cs_hint=cs_hint)
        elif state in ("found", "elsewhere"):
            doi_valid = True
    if doi_valid and best:
        best = _forgive_doi(best)
        if best.status == "verified":
            best.cs_hint = cs_hint
            return best

    # Sudah dicari di SEMUA sumber dalam beberapa jam terakhir dan tidak terverifikasi: jangan hujani API lagi.
    if await asyncio.to_thread(kb.recently_missed, ref.title, settings.negative_cache_hours):
        if best:
            best.cs_hint = cs_hint
            return best
        return Evidence(status="not_found")

    # 2b. API eksternal, berurutan; berhenti pada sumber pertama yang memverifikasi
    reachable = 0  # jumlah API yang berhasil dihubungi (apa pun hasilnya)
    for name, search in PROVIDERS:
        try:
            records = await search(client, ref)
        except Exception as exc:  # noqa: BLE001 - satu API bermasalah tidak boleh mematikan pipeline
            log.warning("Sumber %s gagal untuk %.60r: %s: %s", name, ref.title, type(exc).__name__, exc)
            continue
        reachable += 1
        matches = [compare(ref, rec, name, cached=False) for rec in records]
        found = best_of(matches)
        if found is None:
            continue
        cs_hint = cs_hint or any(m.cs_hint for m in matches)
        if doi_valid:
            found = _forgive_doi(found)
        await asyncio.to_thread(kb.insert, found.record)  # cache: rekaman nyata disimpan agar berikutnya cukup dari lokal
        if found.status == "verified":
            found.cs_hint = cs_hint
            return found
        if best is None or _score(found) > _score(best):
            best = found

    if reachable == 0 and best is None:  # semua API gagal: bukan bukti apa pun, jangan dicatat sebagai "sudah dicari"
        return Evidence(status="not_found", unreachable=True)
    await asyncio.to_thread(kb.remember_miss, ref.title)
    if best is None:
        return Evidence(status="not_found")
    best.cs_hint = cs_hint
    return best
