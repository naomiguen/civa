"""Utilitas logging kecil."""
from __future__ import annotations

import logging


class _DropConnectionLost(logging.Filter):
    """Buang HANYA pesan 'Exception in callback ..._call_connection_lost()' (WinError 10054) dari asyncio.

    Itu kebisingan tak berbahaya pada Windows/Proactor saat browser menutup/menyegarkan koneksi; error asyncio
    lain tetap ditampilkan.
    """

    tag = "civa-drop-connection-lost"

    def filter(self, record: logging.LogRecord) -> bool:
        return "_call_connection_lost" not in record.getMessage()


def silence_connection_reset() -> None:
    """Pasang penyaring pada logger 'asyncio' (aman dipanggil berulang, mis. tiap rerun Streamlit)."""
    logger = logging.getLogger("asyncio")
    if not any(getattr(f, "tag", "") == _DropConnectionLost.tag for f in logger.filters):
        logger.addFilter(_DropConnectionLost())
