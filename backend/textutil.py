"""Fungsi teks murni (tanpa I/O): normalisasi kunci, kemiripan judul, kecocokan penulis, DOI, tahun, pemecah referensi."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional, Sequence

from rapidfuzz import fuzz

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TITLE_SEGMENT_RE = re.compile(r"\s*[:;?!.]\s+|\s+[-–—]+\s+")  # pemisah judul utama / subjudul
_ET_AL_RE = re.compile(r"\bet\.?\s*al\b|\bdkk\b|\.\.\.|…", re.IGNORECASE)  # daftar penulis sengaja dipersingkat
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DBLP_SUFFIX_RE = re.compile(r"\s+\d{4}$")  # DBLP membedakan nama kembar: "Wei Zhang 0001"
_MARKER_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s+")
_HEADING_RE = re.compile(r"^\s*(daftar pustaka|references?|bibliography|referensi)\s*:?\s*$", re.IGNORECASE)
# Awal referensi gaya APA/Harvard: "Ulhaq, N.D. ..." / "Van der Berg, J. ..." (nama belakang, lalu inisial)
_AUTHOR_START_RE = re.compile(
    r"^[^\W\d_][\w'’\-.]*(?:\s+[^\W\d_][\w'’\-.]*){0,3},\s*(?:[^\W\d_]\.\s?){1,4}")
_ENDS_WITH_INITIALS_RE = re.compile(r",\s*(?:[A-ZÀ-Þ]\.\s?){1,4}$")  # daftar penulis masih berlanjut
_TERMINATOR_RE = re.compile(r"(?:[.)\]\"”]|https?://\S+)$")


# --------------------------------------------------------------------------------------
# Judul, DOI, tahun
# --------------------------------------------------------------------------------------
def norm_key(text: object) -> str:
    """Kunci pembanding: huruf kecil, tanpa aksen/tanda baca, spasi tunggal."""
    text = unicodedata.normalize("NFKD", "" if text is None else str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_ALNUM_RE.sub(" ", text).strip()


def title_similarity(a: object, b: object) -> float:
    """Kemiripan dua judul, 0..1. Toleran urutan kata dan subjudul yang terpotong."""
    na, nb = norm_key(a), norm_key(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    score = max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb)) / 100.0
    short, long_ = sorted((na, nb), key=len)
    if len(short) >= 25 and long_.startswith(short + " "):  # mis. subjudul setelah ':' tidak ditulis
        score = max(score, 0.93)
    return score


def _title_words(text: object) -> list[str]:
    return norm_key(str(text or "").replace("&", " and ")).split()


def title_word_issue(reference: object, record: object) -> Optional[str]:
    """Pola manipulasi kata judul referensi terhadap judul aslinya (title_similarity sengaja toleran urutan kata).

    "urutan judul" : kata-katanya sama persis, tetapi urutannya diacak
    "kata judul"   : semua kata referensi ada di judul asli, tetapi ada kata asli yang dihilangkan
    None           : tidak ada pola tersebut. Menghilangkan judul utama/subjudul secara utuh (bagian yang dipisah
                     ':' '?' '.' atau ' - ') dan kata sandang "a/an/the" di awal tetap dianggap wajar.
    """
    ref, rec = _title_words(reference), _title_words(record)
    if not ref or ref == rec:
        return None
    if sorted(ref) == sorted(rec):
        return "urutan judul"
    if not Counter(ref) - Counter(rec):  # semua kata referensi berasal dari judul asli -> ada yang dihilangkan
        if all(word.isdigit() for word in (Counter(rec) - Counter(ref)).elements()):
            return None  # hanya angka (mis. tahun di ujung judul) yang hilang
        allowed = {" ".join(rec[1:])} if rec[0] in ("a", "an", "the") else set()
        segments = [" ".join(_title_words(s)) for s in _TITLE_SEGMENT_RE.split(str(record)) if _title_words(s)]
        allowed |= {" ".join(segments[i:j]) for i in range(len(segments)) for j in range(i + 1, len(segments) + 1)}
        return None if " ".join(ref) in allowed else "kata judul"
    joined_ref, joined_rec = " ".join(ref), " ".join(rec)
    if fuzz.token_sort_ratio(joined_ref, joined_rec) >= 90 and fuzz.ratio(joined_ref, joined_rec) < 75:
        return "urutan judul"  # kata-katanya hampir sama tetapi diacak (tokenisasi sedikit beda, mis. "V4+/V5+")
    return None


def contains_title(raw: object, title: object, min_words: int = 4) -> bool:
    """Judul (min. `min_words` kata) tertulis utuh dan berurutan di teks referensi mentah.

    Dipakai untuk membedakan judul yang memang dimanipulasi dari judul yang terpotong saat diekstraksi GROBID.
    """
    words = _title_words(title)
    return len(words) >= min_words and f" {' '.join(words)} " in f" {' '.join(_title_words(raw))} "


_DASHES_RE = re.compile("[‐-―−﹘﹣－]")  # berbagai "tanda hubung" hasil salin PDF/Word
_INVISIBLE_RE = re.compile("[­​-‏⁠﻿]")  # soft-hyphen, zero-width, BOM


def normalize_doi(value: object) -> Optional[str]:
    """'https://doi.org/10.1000/ABC' -> '10.1000/abc' ; None bila bukan DOI.

    Tahan terhadap artefak salin-tempel: tanda hubung Unicode (non-breaking hyphen, en dash, dst.) dan karakter
    tak terlihat diubah menjadi '-' / dibuang, sehingga DOI dari PDF/Word tetap identik dengan DOI aslinya.
    """
    if value is None:
        return None
    text = _DASHES_RE.sub("-", _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", str(value))))
    match = _DOI_RE.search(text)
    return match.group(0).rstrip(".,;:)]}\"'”’").lower() if match else None


_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*|10\.48550/arxiv\.)"
    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})", re.IGNORECASE)
_IDENTIFIER_NOISE_RE = re.compile(r"https?://\S+|\bdoi\s*:?|10\.\d{4,9}/\S+|arxiv\s*:?\s*\S+", re.IGNORECASE)


def extract_arxiv_id(text: object) -> Optional[str]:
    """'https://arxiv.org/abs/1706.03762v7' | 'arXiv:1706.03762' | '10.48550/arXiv.1706.03762' -> '1706.03762'."""
    match = _ARXIV_ID_RE.search("" if text is None else str(text))
    return match.group(1) if match else None


def is_identifier_only(raw: str) -> bool:
    """Teks hanya berisi DOI / link arXiv (tanpa judul): sisa huruf di luar pengenal kurang dari 12."""
    if not (normalize_doi(raw) or extract_arxiv_id(raw)):
        return False
    rest = _IDENTIFIER_NOISE_RE.sub(" ", raw)
    return len(re.findall(r"[^\W\d_]", rest)) < 12


_PAREN_YEAR_RE = re.compile(r"\(\s*((?:1[89]|20)\d{2})[a-z]?\s*\)")


def paren_year(text: object) -> Optional[int]:
    """Tahun dalam tanda kurung pertama pada teks referensi gaya APA/Harvard: '(2025)' atau '(2025a)'."""
    match = _PAREN_YEAR_RE.search("" if text is None else str(text))
    return int(match.group(1)) if match else None


_QUOTED_TITLE_RE = re.compile(r"[“\"]([^”\"]{10,})[”\"]")


def title_from_raw(raw: object, has_authors: bool) -> str:
    """Judul cadangan bila GROBID tidak menemukannya; '' bila pola tidak dikenali.

    Gaya yang dikenali: judul di antara tanda kutip (Harvard Indonesia); APA "Penulis. (2020). Judul. Sumber." dan
    APA tanpa penulis "Judul. (2020). Sumber.". `has_authors`: GROBID menemukan penulis (judul ada SETELAH tahun).
    """
    text = _MARKER_RE.sub("", _IDENTIFIER_NOISE_RE.sub(" ", "" if raw is None else str(raw))).strip()
    quoted = _QUOTED_TITLE_RE.search(text)
    if quoted:
        return quoted.group(1).strip(" ,.")
    match = _PAREN_YEAR_RE.search(text)
    if not match:
        return ""
    before = text[:match.start()].strip(" .")
    if not has_authors and len(before.split()) >= 4 and not _AUTHOR_START_RE.match(before):
        return before
    after = text[match.end():].lstrip(" .")
    return re.split(r"(?<=[^\s.])\.\s|\.$", after, maxsplit=1)[0].strip()


_TAIL_REPLACEMENTS = ((re.compile(r"\bTersedia pada\s*:?", re.IGNORECASE), "Available at:"),
                      (re.compile(r"\bhal\.", re.IGNORECASE), "pp."),
                      (re.compile(r"\bdiakses\b", re.IGNORECASE), "accessed"))


def _is_caps_token(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def titlecase_caps_runs(text: str, min_run: int = 3) -> str:
    """Ubah rangkaian >= `min_run` kata HURUF KAPITAL berurutan menjadi 'Kapital Awal' (khusus untuk GROBID).

    GROBID memakai pola kapitalisasi sebagai fitur utama dan mengembalikan judul KOSONG untuk judul ALL-CAPS
    (umum pada jurnal Indonesia). Teks asli pengguna tidak diubah; lihat restore_title_case().
    """
    out: list[str] = []
    run: list[str] = []  # token + spasi yang sedang dikumpulkan

    def flush() -> None:
        words = [t for t in run if t.strip()]
        out.extend((t.capitalize() if t.strip() else t) if len(words) >= min_run else t for t in run)
        run.clear()

    for token in re.split(r"(\s+)", text):
        if token.strip() and _is_caps_token(token):
            run.append(token)
        elif not token.strip() and run:
            run.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return "".join(out)


def restore_title_case(title: str, raw: str) -> str:
    """Kembalikan judul ke kapitalisasi ASLI pada teks pengguna (mis. judul yang diubah hanya untuk GROBID)."""
    words = title.split()
    if not words:
        return title
    match = re.search(r"\s+".join(re.escape(w) for w in words), raw, flags=re.IGNORECASE)
    return match.group(0) if match else title


def prepare_for_grobid(raw: str) -> str:
    """Terjemahkan kata penghubung/penanda Indonesia yang membingungkan GROBID (dilatih pada referensi Inggris).

    'dan' antar-penulis -> '&', 'dkk.' -> 'et al.', 'Tersedia pada:' -> 'Available at:', 'hal.' -> 'pp.'.
    Hanya mengubah teks yang DIKIRIM ke GROBID; teks asli pengguna tetap utuh.
    """
    raw = titlecase_caps_runs(raw)
    match = re.search(r"\(\s*(?:1[89]|20)\d{2}[a-z]?\s*\)", raw)
    if not match:
        return raw
    head, tail = raw[:match.start()], raw[match.start():]
    head = re.sub(r"\s+dan\s+", " & ", head, flags=re.IGNORECASE)
    head = re.sub(r"\bdkk\.?", "et al.", head, flags=re.IGNORECASE)
    for pattern, replacement in _TAIL_REPLACEMENTS:
        tail = pattern.sub(replacement, tail)
    return head + tail


def parse_year(value: object) -> Optional[int]:
    match = _YEAR_RE.search("" if value is None else str(value))
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------------------
# Penulis
# --------------------------------------------------------------------------------------
def full_name_tokens(name: str) -> list[str]:
    """Token nama (huruf kecil, tanpa aksen) dari satu nama lengkap; sufiks nomor DBLP dibuang."""
    return [t for t in norm_key(_DBLP_SUFFIX_RE.sub("", name.strip())).split() if t]


def _token_equal(a: str, b: str) -> bool:
    return a == b or (len(a) >= 5 and len(b) >= 5 and fuzz.ratio(a, b) >= 85)  # toleransi salah ketik/transliterasi


def _same_person(given: str, family: str, tokens: list[str]) -> bool:
    """Penulis referensi (given, family) cocok dengan satu nama rekaman (daftar token)?

    Nama belakang harus ada di token rekaman (urutan nama bebas). Bila referensi memberi nama depan/inisial,
    huruf pertamanya harus cocok dengan salah satu token LAIN pada nama yang sama -> 'Smith, J.' != 'Smith, A.'.
    """
    parts = [p for p in family.split() if len(p) >= 2] or family.split()
    position = next((i for i, tok in enumerate(tokens) if any(_token_equal(p, tok) for p in parts)), None)
    if position is None:
        return False
    if not given:
        return True
    others = tokens[:position] + tokens[position + 1:]
    return not others or any(tok[0] == given[0] for tok in others)


def author_overlap(reference_authors: Sequence[tuple[str, str]], record_authors: Iterable[str]) -> Optional[float]:
    """Porsi penulis referensi (given, family) yang ditemukan pada daftar penulis rekaman, 0..1.

    None = tidak bisa dibandingkan (salah satu sisi tidak punya penulis).
    """
    wanted = [(norm_key(g), norm_key(f)) for g, f in reference_authors if norm_key(f)]
    people = [tokens for tokens in (full_name_tokens(a) for a in record_authors) if tokens]
    if not wanted or not people:
        return None
    hits = sum(any(_same_person(g, f, tokens) for tokens in people) for g, f in wanted)
    return hits / len(wanted)


def _name_status(given: str, family: str, tokens: list[str]) -> Optional[str]:
    """Hubungan satu penulis referensi (given, family) dengan satu nama rekaman (token, urutan "Depan Belakang").

    "ok"         : nama belakangnya benar (token terakhir) dan inisialnya cocok; atau nama tengah + inisial cocok
                   (mis. nama belakang pertama orang Spanyol: "Pérez, J." untuk José Pérez García)
    "terbalik"   : nama DEPAN rekaman ditulis sebagai nama belakang ("Walter, W." / "Listia, S. R." untuk Listia Sari)
    "nama_depan" : hanya nama depan/tengah yang ditulis, nama belakangnya hilang ("Panita" untuk Singgih Panita Budiman)
    None         : bukan orang ini
    """
    parts = [p for p in family.split() if len(p) >= 2] or family.split()
    positions = {i for i, tok in enumerate(tokens) if any(_token_equal(p, tok) for p in parts)}
    if not positions:
        return None
    if len(tokens) > 1 and len(tokens) - 1 not in positions:
        if not given:
            return "nama_depan"
        if positions == {0}:
            return "terbalik"
    return "ok" if _same_person(given, family, tokens) else None


def compare_authors(reference_authors: Sequence[tuple[str, str]], record_authors: Iterable[str],
                    raw: str = "") -> tuple[Optional[bool], Optional[str]]:
    """(cocok?, kolom bermasalah) penulis referensi vs rekaman asli. (None, None) = rekaman tanpa penulis.

    Kolom bermasalah: "tanpa penulis" (referensi tidak mencantumkan penulis sama sekali) | "penulis" (sebagian besar
    tidak cocok) | "susunan nama" (nama depan ditulis sebagai nama belakang) | "nama penulis" (hanya nama depan yang
    ditulis) | "jumlah penulis" (ada penulis asli yang dihilangkan tanpa "et al."/"dkk.").
    """
    people = [tokens for tokens in (full_name_tokens(a) for a in record_authors) if tokens]
    wanted = [(norm_key(g), norm_key(f)) for g, f in reference_authors if norm_key(f)]
    if not people:
        return None, None
    if not wanted:
        return False, "tanpa penulis"
    statuses, covered, order = [], set(), []  # status per penulis referensi; orang rekaman tercakup; urutannya
    for given, family in wanted:
        found = {i: s for i, s in ((i, _name_status(given, family, tokens)) for i, tokens in enumerate(people)) if s}
        covered |= found.keys()
        kinds = set(found.values())
        statuses.append("ok" if "ok" in kinds else kinds.pop() if kinds else "hilang")
        if "ok" in kinds:
            order.append(min(i for i, s in found.items() if s == "ok"))
    if statuses.count("hilang") > len(wanted) / 2:
        return False, "penulis"
    if statuses.count("terbalik") and statuses.count("terbalik") >= len(wanted) / 2:
        return False, "susunan nama"
    if "nama_depan" in statuses:
        return False, "nama penulis"
    # Daftar dipersingkat secara wajar: >= 2 penulis PERTAMA, berurutan, tanpa lompatan (banyak gaya selain APA begitu).
    truncated = len(order) == len(wanted) >= 2 and order == list(range(len(order)))
    # Selain itu, penulis asli yang tidak tercakup = dihilangkan (mis. tinggal satu nama, ada yang dilompati, atau dua
    # nama referensi menunjuk orang yang sama).
    if len(covered) < min(len(people), 20) and not truncated and not _ET_AL_RE.search(raw or ""):
        return False, "jumlah penulis"
    return True, None


# --------------------------------------------------------------------------------------
# Pemecah daftar pustaka
# --------------------------------------------------------------------------------------
_OPEN_URL_END_RE = re.compile(r"(?:https?://\S*|10\.\d{4,9}/\S*)[-/_]$", re.IGNORECASE)


def _join_wrapped(parts: list[str]) -> str:
    """Gabungkan baris-baris satu referensi. URL/DOI yang terpotong di akhir baris disambung TANPA spasi."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out = out + part if out and _OPEN_URL_END_RE.search(out) else f"{out} {part}".strip()
    return out


def _terminated(line: str) -> bool:
    """Baris tampak mengakhiri sebuah referensi (titik/kurung/kutip/URL), bukan berakhir pada inisial penulis."""
    line = line.rstrip()
    return bool(_TERMINATOR_RE.search(line)) and not _ENDS_WITH_INITIALS_RE.search(line)


def _author_start_lines(lines: list[str]) -> set[int]:
    """Indeks baris yang tampak membuka referensi baru: diawali 'Nama, I.' dan baris sebelumnya sudah berakhir."""
    return {i for i, ln in enumerate(lines)
            if _AUTHOR_START_RE.match(ln) and (i == 0 or _terminated(lines[i - 1]))}


def split_references(text: str) -> list[str]:
    """Pecah teks daftar pustaka yang ditempel pengguna menjadi referensi individual.

    Mendukung: dipisah baris kosong; bernomor ("1." / "[1]" / "(1)" / "1)") dengan baris lanjutan;
    atau satu referensi per baris. Nomor di awal dibuang.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) > 1:
        chunks = [" ".join(_join_wrapped(block.split("\n")).split()) for block in blocks]
    else:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        author_starts = _author_start_lines(lines)
        if sum(bool(_MARKER_RE.match(ln)) for ln in lines) >= 2:  # bernomor: baris tanpa nomor = lanjutan
            chunks: list[str] = []
            for ln in lines:
                if _MARKER_RE.match(ln) or not chunks:
                    chunks.append(ln)
                else:
                    chunks[-1] = _join_wrapped([chunks[-1], ln])
        elif len(author_starts) >= 2:  # tanpa nomor, tiap referensi terpotong beberapa baris (hard-wrap)
            chunks = []
            for i, ln in enumerate(lines):
                if i == 0 or i in author_starts:
                    chunks.append(ln)
                else:
                    chunks[-1] = _join_wrapped([chunks[-1], ln])
        else:
            chunks = lines
    cleaned = [_MARKER_RE.sub("", chunk, count=1).strip() for chunk in chunks]
    return [c for c in cleaned if c and not _HEADING_RE.match(c)]
