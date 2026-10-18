# CIVA - Citation Intelligence & Verification Assistant

Sistem verifikasi daftar pustaka berbasis Machine Learning untuk mendeteksi sitasi halusinasi pada literatur
Ilmu Komputer. Frontend **Streamlit**, backend **FastAPI**, ekstraksi **GROBID** (Docker), klasifikasi
**MultiBERT + XGBoost / Random Forest / Logistic Regression**.

## Alur sistem

```
teks daftar pustaka
   │
   ├─ TAHAP 1  GROBID  POST /api/processCitation  → <title>, <author> (+ tahun, DOI, jenis)
   ├─ TAHAP 2  Pemeriksaan DOI langsung (Crossref /works/{doi}, registri doi.org) bila DOI dicantumkan, lalu
   │            Retrieval + caching
   │            DB lokal (SQLite) → Crossref → arXiv → OpenAlex → DBLP   (berurutan, berhenti saat terverifikasi)
   │            temuan dari API eksternal di-INSERT ke DB lokal (berikutnya cukup dari lokal)
   ├─ TAHAP 3  ML MURNI TEKS: embedding MultiBERT judul (384) + author (384) = 768-d → classifier → P(valid)
   ├─ TAHAP 4  Filter domain (Semantic Thresholding): di luar Ilmu Komputer → "DI LUAR SCOPE"
   └─ KEPUTUSAN  P(valid) ML + bukti pangkalan data (logit) → VALID / HALU + "Alasan"
```

Model ML **tidak pernah** memakai fitur angka perbandingan, similarity, atau jarak ke database; hanya teks
judul dan author. Bukti Tahap 2 hanya dipakai setelah model, pada lapisan keputusan (lihat `backend/verdict.py`).

## Menjalankan (berurutan, terminal di `C:\Users\Administrator\civa`)

**0. Prasyarat:** Python 3.11+ (diuji 3.13), Docker Desktop.

**1. Lingkungan Python**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `Activate.ps1` diblokir PowerShell? Jalankan dulu `Set-ExecutionPolicy -Scope Process Bypass`.
>
> Python global Anda sudah punya torch/sentence-transformers? Hemat unduhan dengan
> `python -m venv .venv --system-site-packages`. Bila `pip` gagal dengan `WinError 32` pada paket besar, ulangi
> perintahnya (biasanya antivirus mengunci berkas sementara).

**2. Jalankan GROBID** (terminal terpisah; biarkan terbuka). Varian `-crf` hanya ±0,5 GB dan cukup untuk
`processCitation` (varian `full`/default 9-14 GB):

```powershell
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-crf
```

Cek: `curl http://localhost:8070/api/isalive` → `true`.

**3. Latih model** (sekali; membaca `dataset\dataset_asli_cs.xlsx` dan `dataset\dataset_hallucination_*.csv`)

```powershell
python train_model.py            # grid penuh, ±8 menit di CPU
python train_model.py --quick    # grid kecil, ±1 menit bila cache embedding sudah ada
```

Menghasilkan di `models\`: `civa_pipeline.joblib` (pipeline MultiBERT → classifier terkalibrasi),
`multibert_tokenizer.joblib`, `domain_filter.joblib`, `training_report.json`, `model_comparison.csv`.
Model MultiBERT (±470 MB) diunduh otomatis dari HuggingFace pada pemakaian pertama.

**4. Jalankan backend** (terminal baru, venv aktif)

```powershell
uvicorn backend.main:app --port 8000
```

Saat pertama kali jalan, database lokal `data\civa.db` terisi otomatis dari dataset asli (±6.000 referensi).
Cek: `http://localhost:8000/health` · dokumentasi API: `http://localhost:8000/docs`.

**5. Jalankan frontend** (terminal baru, venv aktif)

```powershell
streamlit run frontend/app.py
```

Pertama kali, Streamlit menampilkan prompt onboarding `Email:`; cukup tekan **Enter** (kosongkan), prompt tidak
muncul lagi. Buka `http://localhost:8501`. Titik di header: hijau = backend, model, dan GROBID siap; kuning = ada yang belum siap
(arahkan kursor untuk detail); merah = backend tidak dapat dihubungi.

**6. (Opsional) Uji unit**

```powershell
python -m unittest discover -s tests -v
```

## Struktur proyek

```
civa/
├─ train_model.py        pelatihan (fitur teks 768-d, perbandingan LR/RF/XGBoost, kalibrasi, simpan .joblib)
├─ civa_core.py          preprocessing + MultiBERTEmbedder + DomainFilter (dipakai training DAN backend)
├─ backend/
│  ├─ main.py            FastAPI: alur Tahap 1-4 + keputusan
│  ├─ grobid_client.py   klien GROBID + parser TEI (abaikan namespace)
│  ├─ retrieval.py       DB lokal → Crossref → arXiv → OpenAlex → DBLP, pencocokan metadata
│  ├─ database.py        SQLite: knowledge base + cache hasil API + cache "tidak ditemukan"
│  ├─ verdict.py         penggabungan P(valid) + bukti, teks "Alasan"
│  ├─ textutil.py        kemiripan judul, kecocokan penulis, pemecah referensi
│  ├─ schemas.py · config.py
├─ frontend/app.py       Streamlit (mengikuti desain UI/UX)
├─ tests/test_core.py
├─ dataset/              data masukan
└─ models/ · data/       hasil pelatihan · database lokal (dibuat otomatis)
```

## API

`POST /api/verify` dengan `{"text": "<daftar pustaka, maks. 10 referensi>"}` mengembalikan:

```jsonc
{
  "summary": {"total": 5, "valid": 3, "halu": 1, "out_of_scope": 1, "unparsed": 0, "valid_percent": 60.0},
  "results": [{
    "title": "...", "authors_display": "Zhang, Y. & Chen, L.", "year": 2023, "type": "Jurnal",
    "status": "valid|halu|out_of_scope|unparsed", "label": "VALID", "confidence": 0.95, "reason": "Alasan: ...",
    "ml":       {"label": "valid", "probability_valid": 0.72, "probability_halu": 0.28},   // murni teks
    "evidence": {"status": "verified|mismatch|not_found", "source": "local|crossref|arxiv|openalex|dblp", ...},
    "domain":   {"knn_score": 0.79, "margin": 0.43, "in_scope": true}
  }]
}
```

## Konfigurasi (environment variable, semuanya opsional)

| Variabel | Default | Fungsi |
|---|---|---|
| `CIVA_GROBID_URL` | `http://localhost:8070` | alamat GROBID |
| `CIVA_API_URL` | `http://localhost:8000` | alamat backend yang dipakai frontend |
| `CIVA_CONTACT_EMAIL` | kosong | email Anda untuk "polite pool" Crossref/OpenAlex (tidak diisi otomatis) |
| `CIVA_EVIDENCE_WEIGHT` | `4.0` | pergeseran logit untuk bukti `verified` (+) / `mismatch` (−) |
| `CIVA_NOT_FOUND_WEIGHT` | `-1.5` | pergeseran logit bila tidak ditemukan di mana pun |
| `CIVA_TITLE_MATCH_THRESHOLD` | `0.90` | kemiripan judul minimum agar dianggap ditemukan |
| `CIVA_MISMATCH_TITLE_THRESHOLD` | `0.97` | kemiripan judul minimum untuk menuduh "penulis/tahun tidak sesuai" |
| `CIVA_DOI_CONFLICT_TITLE_THRESHOLD` | `0.60` | DOI sah milik makalah lain hanya dituduh bila judul jelas berbeda |
| `CIVA_GROBID_TIMEOUT` | `90` | detik menunggu jawaban GROBID per percobaan (diulang 3x) |
| `CIVA_MAX_REFERENCES` | `10` | batas referensi per pemeriksaan |

Set `CIVA_EVIDENCE_WEIGHT=0` dan `CIVA_NOT_FOUND_WEIGHT=0` agar verdict **sepenuhnya** dari model ML.

## Hasil evaluasi (apa adanya)

Data: 5.993 referensi asli + 750 halusinasi (setelah pembersihan), split stratified 80/20, CV 5-fold pada data latih.

| Model | CV ROC-AUC | Uji ROC-AUC | Parameter terbaik |
|---|---|---|---|
| Logistic Regression | 0,742 ± 0,014 | 0,736 | C = 0,003 |
| Random Forest | 0,716 ± 0,020 | 0,727 | min_samples_leaf = 5 |
| XGBoost | 0,735 ± 0,019 | 0,741 | max_depth = 3, 500 pohon |

Model terpilih (CV): Logistic Regression, probabilitas dikalibrasi (Platt, bobot kelas seimbang). Pada data uji:
ROC-AUC 0,74 · PR-AUC kelas halu 0,26 (baseline 0,11) · F1 halu 0,32 · balanced accuracy 0,69.
Ablasi: **judul saja** AUC 0,74; **author saja** 0,57; keduanya 0,74.

**Cara membacanya:** klasifikasi murni-teks pada dataset ini hanya menghasilkan pemisahan yang **lemah**. Judul
halusinasi dibuat meniru gaya judul asli (bahasa, kapitalisasi, panjang sama), sehingga tidak ada jalan pintas;
sinyalnya hampir seluruhnya dari judul, sedangkan author nyaris tidak membantu. Model **tidak dapat memeriksa
keberadaan** sebuah makalah; itu pekerjaan Tahap 2, dan keputusan akhir sebagian besar bertumpu pada bukti tersebut.
Skor di atas mengukur kemampuan membedakan dataset sintetis ini, bukan halusinasi LLM di dunia nyata.

Filter domain: ambang kNN dari persentil ke-25 skor leave-one-out korpus CS; judul CS asli yang salah diisolasi
6,7% (korpus "CS" sendiri memuat sebagian makalah manajemen/komunikasi); pada daftar tulisan tangan kecil, 100%
non-CS terisolasi dan 0% CS salah terisolasi. Referensi yang ditemukan di sumber CS (dataset seed, DBLP, arXiv
`cs.*`, OpenAlex "Computer Science") otomatis dianggap in-scope.

## Evaluasi sistem end-to-end

`python evaluate_system.py` mengukur pipeline utuh (GROBID asli + API eksternal + ML + filter domain) pada 196
referensi berlabel, dengan backend & database sementara (tidak menyentuh `data/civa.db`; GROBID harus berjalan).
Laporan lengkap: `reports/system_evaluation.md`, rincian per referensi: `reports/system_eval_results.csv`.

| Ukuran | Pipeline lengkap | Model ML saja |
|---|---|---|
| Recall halusinasi (96 referensi palsu) | **93%** | 54% |
| Precision | **99%** | - |
| Referensi nyata salah dituduh halu | **1%** | 41% |

Kesalahan terbanyak berasal dari filter domain, bukan dari deteksi: 11 referensi "nyata" dinyatakan DI LUAR SCOPE (sebagian
besar memang non-CS: label bidang OpenAlex yang dipakai sebagai kebenaran berisik), dan 6 referensi palsu lolos karena
diisolasi sebagai di luar scope. Dua referensi berjudul HURUF KAPITAL tidak terbaca GROBID. Set uji bersifat sintetis dan
sirkular untuk cakupan indeks (lihat catatan metodologi di laporan); jangan dibaca sebagai akurasi di dunia nyata.

## Catatan

- **Pencarian eksternal** memakai API publik. arXiv dibatasi 1 request / 3 detik, jadi pemeriksaan 10 referensi
  yang tidak ditemukan bisa memakan puluhan detik pada kali pertama; hasil yang ditemukan dan pencarian yang
  gagal di-cache sehingga pemeriksaan ulang hampir seketika.
- **Tidak ditemukan ≠ pasti palsu**: buku, skripsi, dan jurnal lokal sering tidak terindeks. Itu sebabnya
  bobot `not_found` hanya menggeser probabilitas, tidak memutuskan sendiri. Bila semua API eksternal tidak dapat
  dihubungi, bukti ini diabaikan.
- Format input yang dikenali: bernomor (`1.`, `[1]`, `(1)`), dipisah baris kosong, satu referensi per baris, atau tanpa
  nomor tetapi terpotong beberapa baris (awal referensi dikenali dari pola `Nama, I.` setelah referensi sebelumnya
  berakhir). Satu referensi tunggal yang terpotong tanpa nomor dan tanpa baris kosong akan dihitung per baris;
  beri nomor atau satukan menjadi satu baris.
- Boleh menempel **hanya DOI atau link arXiv** (satu per baris), mis. `https://doi.org/10.30598/...` atau
  `https://arxiv.org/abs/1706.03762`. Judul, penulis, dan tahun diambil dari Crossref / DataCite / arXiv; DOI yang
  tidak terdaftar langsung ditandai HALU. Link lain (halaman penerbit, Google Scholar) belum didukung.
- Jumlah referensi pada penghitung "x / 10" diperbarui saat Anda meninggalkan kotak teks atau menekan Ctrl+Enter.
- "Periksa Ulang Teks" kembali ke halaman input dengan teks terakhir terisi, sehingga bisa diedit lalu diperiksa lagi.
- Ikon langkah 4 pada desain tampak terpotong; di sini dipakai ikon filter.
