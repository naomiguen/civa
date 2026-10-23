"""
CIVA - Citation Intelligence & Verification Assistant : FastAPI backend.

Alur verifikasi per referensi
  TAHAP 1  GROBID            ekstrak <title> & <author> (+ tahun, DOI, jenis) dari teks mentah
  TAHAP 2  Retrieval+Cache   database lokal -> Crossref -> arXiv -> OpenAlex -> DBLP (berurutan);
                             temuan dari API eksternal di-INSERT ke database lokal
  TAHAP 3  ML (MURNI TEKS)   embedding MultiBERT judul (384) + author (384) = 768-d -> classifier -> P(valid)
  TAHAP 4  Filter domain     Semantic Thresholding: P(Ilmu Komputer) < ambang -> DI LUAR SCOPE
  KEPUTUSAN                  verdict.fuse(): probabilitas ML + bukti basis data -> VALID / HALU (+ alasan);
                             referensi di luar scope tetap dinilai valid/halu, dengan peringatan untuk dicek ulang

Jalankan (dari folder proyek):  uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

import civa_core
from backend import grobid_client, retrieval, verdict
from backend.config import settings
from backend.database import KnowledgeBase
from backend.logutil import silence_connection_reset
from backend.schemas import (DomainOut, Evidence, EvidenceOut, HealthResponse, MLOut, ParsedReference, Record,
                             RecordOut, ReferenceResult, Summary, VerifyRequest, VerifyResponse)
from backend.textutil import is_identifier_only, split_references, title_from_raw

log = logging.getLogger("civa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
silence_connection_reset()


# --------------------------------------------------------------------------------------
# Mesin: semua komponen yang dimuat sekali saat startup
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        self.kb: Optional[KnowledgeBase] = None
        self.http: Optional[httpx.AsyncClient] = None
        self.embedder: Optional[civa_core.MultiBERTEmbedder] = None
        self.classifier = None
        self.domain: Optional[civa_core.DomainFilter] = None
        self.report: dict = {}
        self._ml_lock = threading.Lock()  # inferensi torch cukup satu per satu (CPU-bound)

    @property
    def ready(self) -> bool:
        return self.classifier is not None and self.domain is not None

    def infer(self, title: str, authors: str) -> tuple[float, float]:
        """TAHAP 3 + 4 untuk satu referensi -> (P(valid), P(Ilmu Komputer)).

        Fitur model = 768 dimensi murni dari TEKS judul + author. Skor domain memakai paruh judul (384 dimensi
        pertama) tetapi tidak pernah masuk ke classifier.
        """
        frame = pd.DataFrame({"judul": [title], "author": [authors]})
        with self._ml_lock:
            features = self.embedder.transform(frame)
            proba = self.classifier.predict_proba(features)[0]
        p_valid = float(proba[list(self.classifier.classes_).index(1)])
        p_cs = self.domain.p_cs(features[:, : civa_core.EMB_DIM])
        return p_valid, float(p_cs[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine()
    engine.kb = await asyncio.to_thread(KnowledgeBase, settings.db_path)
    await asyncio.to_thread(engine.kb.seed_if_needed, settings.seed_dataset)
    agent = "CIVA/1.0 (academic citation verification)"
    if settings.contact_email:
        agent += f" mailto:{settings.contact_email}"
    engine.http = httpx.AsyncClient(headers={"User-Agent": agent}, follow_redirects=True,
                                    limits=httpx.Limits(max_connections=20))
    try:
        pipeline, engine.domain = await asyncio.to_thread(civa_core.load_artifacts)
        engine.embedder, engine.classifier = pipeline.named_steps["embed"], pipeline.named_steps["clf"]
        if civa_core.REPORT_PATH.exists():
            import json
            engine.report = json.loads(civa_core.REPORT_PATH.read_text(encoding="utf-8"))
        await asyncio.to_thread(engine.infer, "warm up", "a b")  # muat MultiBERT sekarang, bukan saat request pertama
        log.info("Model ML siap (%s). Knowledge base: %d referensi.", engine.report.get("best_model", "?"),
                 await asyncio.to_thread(engine.kb.count))
    except FileNotFoundError as exc:
        log.error("%s", exc)
    app.state.engine = engine
    warm_up = asyncio.create_task(grobid_client.warm_up(engine.http))  # latar belakang; tidak menahan startup
    yield
    warm_up.cancel()
    await engine.http.aclose()


app = FastAPI(title="CIVA - Citation Intelligence & Verification Assistant", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------------------
# Pipeline satu referensi
# --------------------------------------------------------------------------------------
def _record_out(record: Optional[Record]) -> Optional[RecordOut]:
    if record is None:
        return None
    data = asdict(record)
    data.pop("id", None)  # id internal database tidak perlu diekspos
    return RecordOut(**data)


def _unparsed(index: int, raw: str, parsed: Optional[ParsedReference], reason: Optional[str] = None) -> ReferenceResult:
    """Referensi yang judulnya tidak bisa diekstraksi: tidak divonis valid/halu."""
    title = (parsed.title or None) if parsed else None
    return ReferenceResult(
        index=index, raw=raw, title=title,
        authors_display=verdict.authors_display(parsed.authors) if parsed else "",
        status="unparsed", label=verdict.STATUS_LABEL["unparsed"],
        reason=reason or verdict.build_reason("unparsed", Evidence()),
    )


def _identifier_missing(index: int, raw: str, parsed: ParsedReference) -> ReferenceResult:
    """Teks hanya berisi DOI dan DOI itu tidak terdaftar di Crossref maupun registri DOI."""
    evidence = Evidence(status="mismatch", source="crossref", doi_match=False, mismatch_fields=["DOI"],
                        doi_issue="missing")
    _, confidence, p_final = verdict.fuse(0.5, evidence, settings.valid_threshold)  # tanpa judul: ML tidak dipakai
    return ReferenceResult(
        index=index, raw=raw, title=None, authors_display="DOI: " + (parsed.doi or ""), doi=parsed.doi,
        type="Lainnya", status="halu", label=verdict.STATUS_LABEL["halu"], confidence=round(confidence, 4),
        halu_probability=round(1 - p_final, 4),
        reason=verdict.build_reason("halu", evidence) + " Periksa juga penulisan DOI-nya.",
        evidence=EvidenceOut(status="mismatch", source="crossref", doi_issue="missing", doi_match=False,
                             mismatch_fields=["DOI"]),
    )


async def verify_reference(index: int, raw: str, engine: Engine, gate: asyncio.Semaphore) -> ReferenceResult:
    async with gate:
        # ---- TAHAP 1: GROBID -> judul & author ------------------------------------------------
        # Teks yang hanya berisi DOI / link arXiv tidak punya judul untuk diekstraksi: lewati GROBID.
        parsed = None if is_identifier_only(raw) else await grobid_client.parse_citation(engine.http, raw)
        if parsed is not None and len(parsed.title.strip()) < 4:
            # GROBID tidak menemukan judul (mis. APA tanpa penulis, atau judul yang kata-katanya diacak): ambil dari
            # teks mentah, agar judul yang DITULIS pengguna yang diperiksa, bukan metadata DOI-nya.
            parsed.title = title_from_raw(raw, bool(parsed.authors))
        if parsed is None or len(parsed.title.strip()) < 4:
            # Tanpa judul: bila ada DOI / link arXiv, ambil metadata dari pengenal itu (Crossref -> DataCite -> arXiv).
            state, from_id, id_record = await retrieval.resolve_identifier(engine.http, raw)
            if state == "found" and from_id and id_record:
                parsed = from_id
                await asyncio.to_thread(engine.kb.insert, id_record)  # cache: Tahap 2 langsung memverifikasi dari lokal
            elif state == "missing" and from_id:
                return _identifier_missing(index, raw, from_id)
            elif state == "unknown":
                return _unparsed(index, raw, parsed, "Alasan: DOI/link tidak dapat diverifikasi saat ini "
                                 "(Crossref, DataCite, atau arXiv tidak menjawab). Coba lagi sebentar.")
            else:
                return _unparsed(index, raw, parsed)

        # ---- TAHAP 2: database lokal -> API eksternal berurutan (+ caching) -----------------------
        evidence = await retrieval.retrieve_evidence(engine.http, engine.kb, parsed)

        # ---- TAHAP 3: ML murni teks   +   TAHAP 4: filter domain ---------------------------------
        author_text = civa_core.authors_from_parts(parsed.authors)
        p_valid, p_cs = await asyncio.to_thread(engine.infer, parsed.title, author_text)
        in_scope = bool(engine.domain.in_scope(np.array([p_cs]))[0])
        forced = False
        if not in_scope and evidence.cs_hint:
            in_scope, forced = True, True  # sumber menyatakan ini literatur Ilmu Komputer (seed/DBLP/arXiv cs./OpenAlex)

    # ---- KEPUTUSAN AKHIR ----------------------------------------------------------------------
    judged, confidence, p_final = verdict.fuse(p_valid, evidence, settings.valid_threshold)
    status, scope_verdict, warning = judged, None, None
    if not in_scope:  # di luar Ilmu Komputer: tetap dinilai valid/halu, tetapi ditandai + peringatan cek ulang
        status, scope_verdict, warning = "out_of_scope", judged, verdict.OOS_WARNING

    record = evidence.record if evidence.status != "not_found" else None
    kind = record.kind if record else parsed.kind
    return ReferenceResult(
        index=index, raw=raw, title=parsed.title,
        authors=[f"{given} {family}".strip() for given, family in parsed.authors],
        authors_display=verdict.authors_display(parsed.authors), year=parsed.year, doi=parsed.doi,
        type=verdict.KIND_LABEL.get(kind, "Lainnya"), status=status, label=verdict.STATUS_LABEL[status],
        scope_verdict=scope_verdict, confidence=round(confidence, 4), halu_probability=round(1 - p_final, 4),
        warning=warning, reason=verdict.build_reason(judged, evidence, parsed.from_identifier),
        ml=MLOut(label="valid" if p_valid >= 0.5 else "halu", probability_valid=round(p_valid, 4),
                 probability_halu=round(1 - p_valid, 4)),
        domain=DomainOut(p_cs=round(p_cs, 4), threshold=round(engine.domain.threshold, 4), in_scope=in_scope,
                         forced_by_knowledge_base=forced),
        evidence=EvidenceOut(
            status=evidence.status, source=evidence.source, cached=evidence.cached, unreachable=evidence.unreachable,
            doi_issue=evidence.doi_issue,
            title_similarity=None if evidence.title_similarity is None else round(evidence.title_similarity, 4),
            author_match=evidence.author_match, year_match=evidence.year_match, doi_match=evidence.doi_match,
            mismatch_fields=evidence.mismatch_fields, record=_record_out(record),
        ),
    )


# --------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine: Engine = request.app.state.engine
    grobid_ok = await grobid_client.is_alive(engine.http)
    return HealthResponse(
        status="ok" if engine.ready and grobid_ok else "degraded", model_loaded=engine.ready,
        grobid_available=grobid_ok, grobid_url=settings.grobid_url,
        knowledge_base_records=await asyncio.to_thread(engine.kb.count),
        model_name=civa_core.MODEL_NAME if engine.ready else None, best_classifier=engine.report.get("best_model"),
    )


@app.post("/api/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest, request: Request) -> VerifyResponse:
    engine: Engine = request.app.state.engine
    if not engine.ready:
        raise HTTPException(503, "Model ML belum tersedia. Jalankan dulu:  python train_model.py")

    references = split_references(body.text)
    if not references:
        raise HTTPException(422, "Teks daftar pustaka kosong.")
    if len(references) > settings.max_references:
        raise HTTPException(422, f"Maksimal {settings.max_references} referensi per pemeriksaan "
                                 f"(terdeteksi {len(references)}).")

    started = time.perf_counter()
    gate = asyncio.Semaphore(settings.concurrency)
    tasks = [asyncio.create_task(verify_reference(i, raw, engine, gate)) for i, raw in enumerate(references, 1)]
    try:
        results = await asyncio.gather(*tasks)
    except grobid_client.GrobidUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except grobid_client.GrobidError as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        for task in tasks:
            task.cancel()  # no-op bila sudah selesai; menghentikan sisa pekerjaan bila ada yang gagal

    counts = Counter(r.status for r in results)
    oos = Counter(r.scope_verdict for r in results if r.status == "out_of_scope")
    overall_valid, overall_halu = verdict.overall_percentages(counts["valid"] + oos["valid"],
                                                              counts["halu"] + oos["halu"])
    return VerifyResponse(
        summary=Summary(total=len(results), valid=counts["valid"], halu=counts["halu"],
                        out_of_scope=counts["out_of_scope"], out_of_scope_valid=oos["valid"],
                        out_of_scope_halu=oos["halu"], unparsed=counts["unparsed"],
                        valid_percent=round(100 * counts["valid"] / len(results), 1),
                        overall_valid_percent=overall_valid, overall_halu_percent=overall_halu),
        results=list(results), elapsed_seconds=round(time.perf_counter() - started, 2),
    )


if __name__ == "__main__":  # alternatif: python -m backend.main
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
