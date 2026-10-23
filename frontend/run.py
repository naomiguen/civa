"""
Jalankan frontend Streamlit dengan terminal yang bersih.

Sama dengan `streamlit run frontend/app.py`, tetapi penyaring kebisingan WinError 10054 (browser menutup/menyegarkan
koneksi di Windows) dipasang SEBELUM server Streamlit hidup, bukan baru saat halaman pertama dijalankan.

  python frontend/run.py                      # argumen tambahan diteruskan, mis. --server.port 8502
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.logutil import silence_connection_reset  # noqa: E402
from streamlit.web import cli as streamlit_cli  # noqa: E402

if __name__ == "__main__":
    silence_connection_reset()
    sys.argv = ["streamlit", "run", str(ROOT / "frontend" / "app.py"), *sys.argv[1:]]
    sys.exit(streamlit_cli.main())
