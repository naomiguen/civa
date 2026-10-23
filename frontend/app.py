"""
CIVA - Citation Intelligence & Verification Assistant : frontend Streamlit.

Mengonsumsi endpoint FastAPI  POST {CIVA_API_URL}/api/verify  dan GET /health.
Tampilan mengikuti desain UI/UX (capstone.pdf): halaman input, lalu halaman hasil dengan donut chart,
kartu rekapitulasi, dan kartu hasil deteksi per referensi.

Jalankan (dari folder proyek):  streamlit run frontend/app.py
"""
from __future__ import annotations

import html
import math
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # agar `backend.*` dapat diimpor
from backend.config import settings  # noqa: E402  (batas jumlah referensi = satu sumber kebenaran dgn backend)
from backend.logutil import silence_connection_reset  # noqa: E402
from backend.textutil import split_references  # noqa: E402

silence_connection_reset()  # sembunyikan kebisingan WinError 10054 saat tab browser ditutup/di-refresh

API_URL = os.getenv("CIVA_API_URL", "http://localhost:8000").rstrip("/")
MAX_REFS = settings.max_references

PLACEHOLDER = (
    f"Masukkan teks daftar pustaka di sini (Maks {MAX_REFS} referensi)...\n\n"
    "Contoh:\n"
    "1. Zhang, Y. & Chen, L. (2023). Deep Learning for Natural Language Processing: A Review. Journal of AI.\n"
    "2. Smith, J. & Patel, R. (2024). Neural Network Approaches to Citation Verification. Fake Journal of Robotics."
)

st.set_page_config(page_title="CIVA - Citation Intelligence & Verification Assistant", page_icon="🎓",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------------------
# Gaya (token warna & ukuran diambil dari desain)
# --------------------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f4f6f5; --header:#0b2620; --header-line:#123c33; --ink:#1e293b; --ink2:#475569; --ink3:#64748b;
  --line:#e2e8f0; --chip:#f1f5f9; --green:#10b981; --donut-green:#21b873; --mint:#e6f4ea; --red:#ef4444;
  --amber:#f59e0b; --grey:#94a3b8;
}
.stApp{background:var(--bg);font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;color:var(--ink)}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
footer,#MainMenu{display:none !important}
.stMainBlockContainer,.block-container{padding:0 !important;max-width:100% !important}
[data-testid="stVerticalBlock"]{gap:0 !important}
[data-testid="stElementContainer"]{margin:0}
button,textarea,[class^="civa-"],[class*=" civa-"],[class^="civa-"] *,[class*=" civa-"] *,
.st-key-btn_check *,.st-key-btn_recheck *{font-family:'Inter','Segoe UI',system-ui,sans-serif !important}

/* ---------- header ---------- */
.civa-header{height:54px;background:var(--header);border-bottom:1px solid var(--header-line);display:flex;
  align-items:center;justify-content:space-between;padding:0 28px}
.civa-brand{display:flex;align-items:center;gap:12px}
.civa-logo{width:32px;height:32px;border-radius:8px;background:var(--green);display:grid;place-items:center;
  color:var(--header);box-shadow:0 0 10px rgba(16,185,129,.4)}
.civa-logo svg{width:18px;height:18px}
.civa-title{color:#f8fafc;font-weight:700;font-size:16.5px}
.civa-status{display:flex;align-items:center;gap:8px;color:#f8fafc;font-size:13.5px;font-weight:500;
  border-left:1px solid var(--header-line);padding-left:14px;height:18px}
.civa-dot{width:7px;height:7px;border-radius:50%;background:var(--green)}
.civa-dot.warn{background:var(--amber)} .civa-dot.down{background:var(--red)}
svg.i{fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ---------- halaman input ---------- */
.st-key-home{max-width:1032px;margin:0 auto;padding:0 16px 80px}
.civa-hero{text-align:center;padding:83px 0 63px}
.civa-hero .h{font-size:40px;line-height:1.2;font-weight:800;color:var(--ink);margin:0 0 16px;padding:0}
.civa-hero p{max-width:896px;margin:0 auto;font-size:17.6px;line-height:1.23;color:var(--ink2)}
.st-key-input_card{background:#fff;border:1px solid var(--line);border-radius:24px;padding:24px 24px 38px;
  box-shadow:0 1px 3px rgba(15,23,42,.05),0 8px 24px -8px rgba(15,23,42,.06)}
.st-key-input_card [data-baseweb="textarea"],.st-key-input_card [data-baseweb="base-input"],
.st-key-input_card [data-testid="stTextAreaRootElement"],.st-key-input_card [data-testid="stTextAreaRootElement"] > div{
  background:transparent !important;border:none !important;box-shadow:none !important;border-radius:0 !important}
.st-key-input_card textarea{background:transparent !important;border:none !important;box-shadow:none !important;
  padding:0 !important;font-size:16.7px !important;line-height:26.4px !important;color:var(--ink) !important;
  resize:none !important;min-height:266px}
.st-key-input_card textarea::placeholder{color:var(--ink3) !important;opacity:1 !important}
.st-key-input_card [data-testid="InputInstructions"]{display:none !important}
.st-key-input_card [data-testid="stTextAreaRootElement"],.st-key-input_card [data-testid="stTextArea"]{margin:0}
.civa-foot{display:flex;justify-content:space-between;align-items:center;font-size:14.4px;color:#000;margin-top:16px}
.civa-foot .over{color:var(--red);font-weight:700}
.st-key-btn_check{display:flex !important;justify-content:center;width:100% !important;margin:46px 0 44px}
.st-key-btn_check [data-testid="stButton"]{display:flex;justify-content:center;width:100%}
.st-key-btn_check button{width:355px;height:58px;border-radius:12px !important;background:var(--green) !important;
  border:none !important;color:#fff !important;font-weight:700;
  box-shadow:0 10px 22px -6px rgba(16,185,129,.45)}
.st-key-btn_check button:hover{background:#0ea371 !important;color:#fff !important}
.st-key-btn_check button p{font-size:17.8px;font-weight:700;color:#fff}
.civa-alert{margin:16px 0 0;padding:14px 18px;border-radius:12px;background:#fef2f2;border:1px solid #fecaca;
  color:#b91c1c;font-size:14px;line-height:1.5;text-align:left}
.civa-rule{border:0;border-top:1px solid var(--line);margin:0 !important}
.civa-h3{font-size:20px;font-weight:700;color:var(--ink);margin:40px 0 24px}
.civa-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:24px}
.civa-step{background:#fff;border:1px solid var(--line);border-radius:16px;padding:24px}
.civa-step .ic{width:44px;height:44px;border-radius:12px;background:var(--mint);color:var(--green);display:grid;
  place-items:center;margin-bottom:18px}
.civa-step .ic svg{width:22px;height:22px}
.civa-step .st{font-size:16.7px;line-height:1.25;font-weight:700;color:var(--ink);margin:0 0 14px;padding:0}
.civa-step p{font-size:14.5px;line-height:1.55;color:var(--ink2);margin:0}

/* ---------- halaman hasil ---------- */
.st-key-result{padding:62px 44px 56px}
.st-key-left_card{background:#fff;border:1px solid var(--line);border-radius:22px;padding:22px 24px 24px;
  box-shadow:0 1px 3px rgba(15,23,42,.05),0 8px 24px -8px rgba(15,23,42,.06)}
.civa-cardtitle{font-size:17.7px;line-height:1.2;font-weight:700;color:var(--ink);margin:0 0 20px;padding:0}
.civa-textbox{background:var(--bg);border-radius:14px;padding:18px;min-height:230px;max-height:340px;overflow:auto;
  font-size:14.4px;line-height:1.55;color:var(--ink2);white-space:pre-wrap;word-break:break-word}
.st-key-btn_recheck{margin:34px 0 18px !important}
.st-key-btn_recheck button{width:100%;height:48px;border-radius:10px !important;background:var(--mint) !important;
  border:1.5px solid var(--green) !important;color:var(--green) !important;font-weight:600}
.st-key-btn_recheck button p{font-size:15.5px;font-weight:600;color:var(--green)}
.st-key-btn_recheck button:hover{background:#d7eee0 !important;border-color:var(--green) !important}
.civa-pctbox{background:var(--bg);border-radius:14px;padding:16px 20px 14px}
.civa-pctbox .t{font-size:15.5px;font-weight:600;color:#666;margin-bottom:13px}
.civa-pie{display:block;width:220px;height:220px;margin:0 auto}
.civa-pie text{text-anchor:middle;fill:var(--ink)}
.civa-pie .n{font-size:12.5px;font-weight:600}
.civa-pie .p{font-size:16px;font-weight:700}
.civa-legend{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-top:16px;font-size:13px;font-weight:600;color:var(--ink2)}
.civa-legend span{display:flex;align-items:center;gap:7px;white-space:nowrap}
.civa-legend i{flex:none;width:11px;height:11px;border-radius:3px}
.civa-legend-note{margin-top:10px;font-size:11.8px;line-height:1.45;color:var(--ink3)}
.civa-right{margin:-4.5px 0 0 6px}
.civa-h2{font-size:20px;font-weight:700;color:var(--ink);margin:0 0 14px;padding:0}
.civa-h2.sp{margin-top:34px}
.civa-stats{display:grid;gap:20px}
.civa-stat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:24px 24px 22px;
  box-shadow:0 1px 3px rgba(15,23,42,.05)}
.civa-stat .l{font-size:15.8px;color:var(--ink2)}
.civa-stat .v{font-size:40px;line-height:1;font-weight:800;margin-top:8px}
.civa-stat .s{font-size:12.8px;font-weight:600;color:var(--ink3);margin-top:8px;white-space:nowrap}
.civa-stat .s b{font-weight:700}
.civa-results{display:flex;flex-direction:column;gap:16px}
.civa-result{background:#fff;border:1px solid rgba(16,185,129,.125);border-radius:14px;padding:20px 20px 24px;
  box-shadow:0 1px 3px rgba(15,23,42,.05)}
.civa-result.bad{border-color:rgba(239,68,68,.12)}
.civa-result .top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.civa-result .ttl{font-size:17.7px;line-height:1.3;font-weight:700;color:var(--ink);word-break:break-word}
.civa-result .meta{display:flex;align-items:center;gap:10px;margin-top:4px;font-size:14.4px;color:var(--ink3)}
.civa-pill{background:var(--chip);color:var(--ink2);font-size:12px;font-weight:700;padding:2px 9px;border-radius:6px}
.civa-badge{flex:none;color:#fff;font-size:14.4px;font-weight:700;padding:6px 14px;border-radius:7px;
  text-transform:uppercase;white-space:nowrap;line-height:1.2}
.civa-result .why{display:flex;align-items:center;gap:9px;margin-top:9px;padding-top:14px;border-top:1px solid var(--line);
  font-size:13.4px;color:var(--ink3);line-height:1.4}
.civa-result .badges{display:flex;align-items:center;gap:8px;flex:none}
.civa-chip{font-size:13px;font-weight:700;padding:5px 10px;border-radius:7px;border:1.5px solid currentColor;
  white-space:nowrap;line-height:1.2;background:#fff}
.civa-result .warn{display:flex;align-items:flex-start;gap:9px;margin-top:12px;padding:10px 12px;border-radius:9px;
  background:#fffbeb;border:1px solid #fde68a;color:#92400e;font-size:12.8px;line-height:1.45}
.civa-result .warn svg{flex:none;width:16px;height:16px;margin-top:1px;color:var(--amber)}
.civa-result .why svg{flex:none;width:16px;height:16px}
.civa-result .why .odot{flex:none;width:16px;height:16px;border-radius:50%}
@media (max-width:900px){.civa-steps{grid-template-columns:repeat(2,1fr)}.civa-header{padding:0 16px}
  .civa-title{font-size:14px}.st-key-result{padding:20px}.st-key-left_card{min-height:0}}
"""

# Ikon (lucide) sebagai SVG inline
ICON = {
    "cap": '<svg class="i" viewBox="0 0 24 24"><path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/><path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/></svg>',
    "file": '<svg class="i" viewBox="0 0 24 24"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
    "shield": '<svg class="i" viewBox="0 0 24 24"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "cpu": '<svg class="i" viewBox="0 0 24 24"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
    "filter": '<svg class="i" viewBox="0 0 24 24"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
    "check": '<svg class="i" viewBox="0 0 24 24" style="color:#10b981"><path d="M20 6 9 17l-5-5"/></svg>',
    "warn": '<svg class="i" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "alert": '<svg class="i" viewBox="0 0 24 24" style="color:#ef4444"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
}

STEPS = [
    ("file", "1. Ekstraksi GROBID",
     "Sistem mengurai teks input daftar pustaka secara otomatis menggunakan model berbasis Machine Learning untuk "
     "mengekstrak entitas metadata penting seperti judul, penulis, dan tahun publikasi."),
    ("shield", "2. Cek DOI &amp; Metadata",
     "Mencocokkan data referensi dengan pangkalan data rujukan internal (Closed-Domain Knowledge Base) yang secara "
     "spesifik dibangun dari sumber dataset akademik terverifikasi seperti Crossref dan arXiv."),
    ("cpu", "3. Klasifikasi Machine Learning",
     "Mengevaluasi hasil ekstraksi fitur referensi menggunakan model klasifikasi terbaik yang dioptimasi dari hasil "
     "pengujian komparatif antara algoritma Random Forest, XGBoost, dan Logistic Regression."),
    ("filter", "4. Filter Domain Otomatis",
     "Menerapkan metode Semantic Thresholding untuk mendeteksi literatur di luar disiplin Ilmu Komputer. Referensi "
     "lintas ilmu tetap dinilai valid/tidak valid, lalu diberi peringatan agar dicek ulang."),
]

STATUS_COLOR = {"valid": "#10b981", "halu": "#ef4444", "out_of_scope": "#f59e0b", "unparsed": "#94a3b8"}


# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------
class ApiError(RuntimeError):
    pass


@st.cache_data(ttl=8, show_spinner=False)
def backend_health() -> tuple[str, str]:
    """(status, tooltip): ok | warn | down  -> warna titik di header."""
    try:
        data = httpx.get(f"{API_URL}/health", timeout=2.0).json()
    except (httpx.HTTPError, ValueError):
        return "down", f"Backend tidak dapat dihubungi di {API_URL}"
    problems = []
    if not data.get("model_loaded"):
        problems.append("model ML belum dilatih")
    if not data.get("grobid_available"):
        problems.append("GROBID tidak aktif")
    if problems:
        return "warn", "Backend aktif, tetapi: " + ", ".join(problems)
    return "ok", f"Backend aktif | KB: {data.get('knowledge_base_records', 0)} referensi"


def call_verify(text: str) -> dict:
    try:
        response = httpx.post(f"{API_URL}/api/verify", json={"text": text},
                              timeout=httpx.Timeout(300.0, connect=5.0))
    except httpx.ConnectError as exc:
        raise ApiError(f"Backend FastAPI tidak dapat dihubungi di {API_URL}. "
                       "Jalankan:  uvicorn backend.main:app --port 8000") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Permintaan ke backend gagal ({type(exc).__name__}).") from exc
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, list):  # galat validasi pydantic
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        raise ApiError(str(detail or f"HTTP {response.status_code}"))
    return response.json()


# --------------------------------------------------------------------------------------
# Komponen HTML
# --------------------------------------------------------------------------------------
def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def header_html() -> str:
    state, tip = backend_health()
    dot = {"ok": "", "warn": " warn", "down": " down"}[state]
    return (f'<div class="civa-header"><div class="civa-brand"><span class="civa-logo">{ICON["cap"]}</span>'
            f'<span class="civa-title">CIVA: Citation Intelligence &amp; Verification Assistant</span></div>'
            f'<div class="civa-status" title="{esc(tip)}"><span class="civa-dot{dot}"></span>FastAPI Backend</div></div>')


def steps_html() -> str:
    cards = "".join(f'<div class="civa-step"><div class="ic">{ICON[icon]}</div><div class="st">{title}</div>'
                    f'<p>{body}</p></div>' for icon, title, body in STEPS)
    return f'<div class="civa-h3">Cara Kerja Sistem</div><div class="civa-steps">{cards}</div>'


def stats_html(summary: dict) -> str:
    cards = [("Total Referensi", summary["total"], "var(--ink)"), ("Valid", summary["valid"], "var(--green)"),
             ("Halusinasi", summary["halu"], "var(--red)"), ("Di Luar Scope", summary["out_of_scope"], "var(--amber)")]
    if summary.get("unparsed"):
        cards.append(("Tidak Terbaca", summary["unparsed"], "var(--grey)"))
    sub = {}
    if summary["out_of_scope"]:  # rincian penilaian referensi di luar scope
        sub["Di Luar Scope"] = (f'<div class="s"><b style="color:var(--green)">{summary.get("out_of_scope_valid", 0)}</b> valid · '
                                f'<b style="color:var(--red)">{summary.get("out_of_scope_halu", 0)}</b> tidak valid</div>')
    body = "".join(f'<div class="civa-stat"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div>'
                   f'{sub.get(label, "")}</div>' for label, value, color in cards)
    return f'<div class="civa-stats" style="grid-template-columns:repeat({len(cards)},1fr)">{body}</div>'


PIE_SLICES = (("valid", "Valid", "#21b873"), ("halu", "Halu", "#ef4444"), ("out_of_scope", "Luar Scope", "#f59e0b"),
              ("unparsed", "Tidak Terbaca", "#cbd5e1"))


def whole_percentages(counts: list[int]) -> list[int]:
    """Persen bulat yang jumlahnya tepat 100 (metode sisa terbesar), mis. [1, 1, 1] -> [34, 33, 33]."""
    raw = [100 * c / sum(counts) for c in counts]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: 100 - sum(out)]:
        out[i] += 1
    return out


def _pie_point(center: float, radius: float, degrees: float) -> tuple[float, float]:
    """Titik pada lingkaran; 0 derajat = arah jam 12, searah jarum jam."""
    rad = math.radians(degrees)
    return center + radius * math.sin(rad), center - radius * math.cos(rad)


def pie_html(summary: dict) -> str:
    """Pie chart penuh: Valid / Halu / Luar Scope (/ Tidak Terbaca), label + persen di dalam irisan."""
    slices = [(label, color, summary.get(key, 0)) for key, label, color in PIE_SLICES if summary.get(key, 0)]
    c, r = 110, 106
    shapes, texts = [], []
    if not slices:
        shapes.append(f'<circle cx="{c}" cy="{c}" r="{r}" fill="#e2e8f0"/>')
    else:
        total, start = sum(n for *_, n in slices), 0.0
        for (label, color, n), pct in zip(slices, whole_percentages([n for *_, n in slices])):
            sweep = 360 * n / total
            if len(slices) == 1:
                shapes.append(f'<circle cx="{c}" cy="{c}" r="{r}" fill="{color}"/>')
            else:
                (x0, y0), (x1, y1) = _pie_point(c, r, start), _pie_point(c, r, start + sweep)
                shapes.append(f'<path d="M{c} {c}L{x0:.2f} {y0:.2f}A{r} {r} 0 {int(sweep > 180)} 1 {x1:.2f} {y1:.2f}Z" '
                              f'fill="{color}" stroke="#fff" stroke-width="2.5" stroke-linejoin="round"/>')
            if pct >= 15:  # nama + persen di dalam irisan
                x, y = _pie_point(c, 0 if len(slices) == 1 else r * 0.6, start + sweep / 2)
                texts.append(f'<text class="n" x="{x:.1f}" y="{y - 3:.1f}">{label}</text>'
                             f'<text class="p" x="{x:.1f}" y="{y + 15:.1f}">{pct}%</text>')
            elif pct >= 5:  # irisan sempit: persen saja, namanya di legenda
                x, y = _pie_point(c, r * 0.68, start + sweep / 2)
                texts.append(f'<text class="p" x="{x:.1f}" y="{y + 6:.1f}">{pct}%</text>')
            start += sweep
    legend = "".join(f'<span><i style="background:{color}"></i>{label} ({summary.get(key, 0)})</span>'
                     for key, label, color in PIE_SLICES if key != "unparsed" or summary.get(key, 0))
    note = ""
    if summary["out_of_scope"]:
        note = (f'<div class="civa-legend-note">Luar scope: {summary.get("out_of_scope_valid", 0)} valid · '
                f'{summary.get("out_of_scope_halu", 0)} tidak valid (perlu dicek ulang).</div>')
    return (f'<div class="civa-pctbox"><div class="t">Persentase Keseluruhan</div>'
            f'<svg class="civa-pie" viewBox="0 0 220 220" role="img">{"".join(shapes)}{"".join(texts)}</svg>'
            f'<div class="civa-legend">{legend}</div>{note}</div>')


def result_card_html(item: dict) -> str:
    status, confidence = item["status"], item.get("confidence")
    if status in ("valid", "halu") and confidence is not None:
        badge = f'{round(confidence * 100)}% {item["label"]}'
    else:
        badge = item["label"]
    title = item.get("title") or item.get("raw", "")
    year = f' ({item["year"]})' if item.get("year") else ""
    pill = f'<span class="civa-pill">{esc(item["type"])}</span>' if status != "unparsed" else ""
    if status == "valid":
        icon = ICON["check"]
    elif status == "halu":
        icon = ICON["alert"]
    else:
        icon = f'<span class="odot" style="background:{STATUS_COLOR[status]}"></span>'
    chip = ""
    if status == "out_of_scope" and item.get("scope_verdict"):  # di luar scope tetap dinilai: valid / tidak valid + % halu
        valid = item["scope_verdict"] == "valid"
        chip = (f'<span class="civa-chip" style="color:{STATUS_COLOR["valid" if valid else "halu"]}" '
                f'title="Di luar scope: perlu dicek ulang">{"VALID" if valid else "TIDAK VALID"} · '
                f'{round(item["halu_probability"] * 100)}% halu</span>')
    warning = ""
    if item.get("warning"):
        warning = f'<div class="warn">{ICON["warn"]}<span>{esc(item["warning"])}</span></div>'
    looks_valid = status == "valid" or item.get("scope_verdict") == "valid"
    return (f'<div class="civa-result{"" if looks_valid else " bad"}"><div class="top"><div>'
            f'<div class="ttl">{esc(title)}</div>'
            f'<div class="meta"><span>{esc(item.get("authors_display", ""))}{esc(year)}</span>{pill}</div></div>'
            f'<div class="badges">{chip}<span class="civa-badge" style="background:{STATUS_COLOR[status]}">{esc(badge)}</span></div></div>'
            f'<div class="why">{icon}<span>{esc(item["reason"])}</span></div>{warning}</div>')


# --------------------------------------------------------------------------------------
# Halaman
# --------------------------------------------------------------------------------------
def run_check(text: str) -> None:
    count = len(split_references(text))
    if count == 0:
        st.session_state["error"] = "Masukkan teks daftar pustaka terlebih dahulu."
    elif count > MAX_REFS:
        st.session_state["error"] = f"Maksimal {MAX_REFS} referensi per pemeriksaan (terdeteksi {count}). Kurangi dulu daftarnya."
    else:
        try:
            with st.spinner("Memverifikasi referensi (GROBID, pangkalan data, model ML)..."):
                st.session_state["result"] = call_verify(text)
            st.session_state.update(saved_text=text, view="result", error=None)
        except ApiError as exc:
            st.session_state["error"] = str(exc)


def home_page() -> None:
    with st.container(key="home"):
        st.markdown('<div class="civa-hero"><div class="h">Verifikasi Keaslian Daftar Pustaka Anda</div>'
                    '<p>Sistem validasi referensi format-agnostic berbasis Machine Learning untuk mendeteksi '
                    'halusinasi sitasi pada literatur Ilmu Komputer.</p></div>', unsafe_allow_html=True)
        with st.container(key="input_card"):
            text = st.text_area("Daftar pustaka", key="refs_text", label_visibility="collapsed",
                                placeholder=PLACEHOLDER, height=268)
            count = len(split_references(text))
            over = ' class="over"' if count > MAX_REFS else ""
            st.markdown(f'<div class="civa-foot"><span>Format: APA, Harvard, dll.</span>'
                        f'<span{over}>{count} / {MAX_REFS} Referensi</span></div>', unsafe_allow_html=True)
        if st.session_state.get("error"):
            st.markdown(f'<div class="civa-alert">{esc(st.session_state["error"])}</div>', unsafe_allow_html=True)
        if st.button("Periksa Daftar Pustaka", key="btn_check", type="primary"):
            run_check(text)
            st.rerun()
        st.markdown('<hr class="civa-rule">' + steps_html(), unsafe_allow_html=True)


def result_page() -> None:
    data = st.session_state["result"]
    summary = data["summary"]
    with st.container(key="result"):
        left, right = st.columns([1, 1.95], gap="medium")
        with left:
            with st.container(key="left_card"):
                st.markdown('<div class="civa-cardtitle">Teks yang Diperiksa</div>'
                            f'<div class="civa-textbox">{esc(st.session_state.get("saved_text", ""))}</div>',
                            unsafe_allow_html=True)
                if st.button("Periksa Ulang Teks", key="btn_recheck", use_container_width=True):
                    st.session_state.update(view="home", refs_text=st.session_state.get("saved_text", ""), error=None)
                    st.rerun()
                st.markdown(pie_html(summary), unsafe_allow_html=True)
        with right:
            cards = "".join(result_card_html(item) for item in data["results"])
            st.markdown('<div class="civa-right"><div class="civa-h2">Rekapitulasi Deteksi</div>' + stats_html(summary)
                        + '<div class="civa-h2 sp">Rincian Hasil Deteksi - CIVA (Citation Intelligence &amp; '
                          'Verification Assistant)</div>' + f'<div class="civa-results">{cards}</div></div>',
                        unsafe_allow_html=True)


def main() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(header_html(), unsafe_allow_html=True)
    if st.session_state.get("view") == "result" and st.session_state.get("result"):
        result_page()
    else:
        home_page()


main()
