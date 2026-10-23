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
   │            bila ditemukan, dicocokkan dgn rekaman asli: kata judul diacak / dihilangkan, penulis berbeda /
   │            tidak dicantumkan / susunan nama terbalik / sebagian dihilangkan, tahun, DOI → "mismatch"
   ├─ TAHAP 3  ML MURNI TEKS: embedding MultiBERT judul (384) + author (384) = 768-d → classifier → P(valid)
   ├─ TAHAP 4  Filter domain (Semantic Thresholding): P(Ilmu Komputer) dari judul < ambang → "DI LUAR SCOPE"
   └─ KEPUTUSAN  P(valid) ML + bukti pangkalan data (logit) → VALID / HALU + "Alasan"
                 referensi DI LUAR SCOPE tetap dinilai (VALID / TIDAK VALID + % halu) dan diberi peringatan cek ulang
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

**3. Latih model** (sekali; membaca `dataset\dataset_scope_6000.csv` dan `dataset\dataset_halu_6000_2kolom.csv`)

```powershell
python train_model.py            # grid penuh, ±10 menit di CPU
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
python frontend/run.py            # sama dengan: streamlit run frontend/app.py
```

`frontend/run.py` memasang penyaring pesan `ConnectionResetError: [WinError 10054]` sebelum server hidup. Pesan itu tidak
berbahaya (muncul di Windows saat tab browser ditutup/di-refresh) dan tetap bisa muncul bila memakai `streamlit run`.

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
  "summary": {"total": 5, "valid": 3, "halu": 1, "out_of_scope": 1, "out_of_scope_valid": 0, "out_of_scope_halu": 1,
              "unparsed": 0, "valid_percent": 60.0,
              "overall_valid_percent": 60.0, "overall_halu_percent": 40.0},   // valid vs halu, termasuk luar scope
  "results": [{
    "title": "...", "authors_display": "Zhang, Y. & Chen, L.", "year": 2023, "type": "Jurnal",
    "status": "valid|halu|out_of_scope|unparsed", "label": "VALID", "confidence": 0.95, "reason": "Alasan: ...",
    "scope_verdict": null,          // untuk out_of_scope: "valid" | "halu" (tetap dinilai)
    "halu_probability": 0.05,       // perkiraan % halu (semua referensi yang terbaca)
    "warning": null,                // untuk out_of_scope: peringatan agar dicek ulang
    "ml":       {"label": "valid", "probability_valid": 0.72, "probability_halu": 0.28},   // murni teks
    "evidence": {"status": "verified|mismatch|not_found", "source": "local|crossref|arxiv|openalex|dblp", ...},
    "domain":   {"p_cs": 0.97, "threshold": 0.43, "in_scope": true}
  }]
}
```

## Konfigurasi (environment variable, semuanya opsional)

| Variabel | Default | Fungsi |
|---|---|---|
| `CIVA_GROBID_URL` | `http://localhost:8070` | alamat GROBID |
| `CIVA_API_URL` | `http://localhost:8000` | alamat backend yang dipakai frontend |
| `CIVA_CONTACT_EMAIL` | kosong | email Anda untuk "polite pool" Crossref/OpenAlex (tidak diisi otomatis) |
| `CIVA_OPENALEX_API_KEY` | kosong | kunci API OpenAlex; tanpa kunci kuota harian dibagi per IP dan bisa habis (429), OpenAlex lalu dilewati |
| `CIVA_EVIDENCE_WEIGHT` | `4.0` | pergeseran logit untuk bukti `verified` (+) / `mismatch` (−) |
| `CIVA_NOT_FOUND_WEIGHT` | `-1.5` | pergeseran logit bila tidak ditemukan di mana pun |
| `CIVA_TITLE_MATCH_THRESHOLD` | `0.90` | kemiripan judul minimum agar dianggap ditemukan |
| `CIVA_MISMATCH_TITLE_THRESHOLD` | `0.97` | kemiripan judul minimum untuk menuduh "penulis/tahun tidak sesuai" |
| `CIVA_DOI_CONFLICT_TITLE_THRESHOLD` | `0.60` | DOI sah milik makalah lain hanya dituduh bila judul jelas berbeda |
| `CIVA_GROBID_TIMEOUT` | `90` | detik menunggu jawaban GROBID per percobaan (diulang 3x) |
| `CIVA_MAX_REFERENCES` | `10` | batas referensi per pemeriksaan |

Set `CIVA_EVIDENCE_WEIGHT=0` dan `CIVA_NOT_FOUND_WEIGHT=0` agar verdict **sepenuhnya** dari model ML.

## Hasil evaluasi (apa adanya)

**Data latih** (`dataset/`):

- `dataset_scope_6000.csv`: 6.000 referensi **asli**, 3.000 Ilmu Komputer + 3.000 dari 23 bidang lain (kedokteran,
  kimia, hukum, ekonomi, dst.; Inggris & Indonesia).
- `dataset_halu_6000_2kolom.csv`: 6.000 referensi **halu** yang semuanya manipulasi dari data asli di atas.
  `train_model.py` menautkan tiap baris halu ke sumbernya (5.999 dari 6.000 tertaut) dan mengenali jenisnya:
  kata judul diacak 1.491, kata judul dihapus 1.490, susunan nama penulis dibalik 1.347, nama penulis tidak
  lengkap / tinggal satu 1.240. **430 baris "halu" identik dengan aslinya** (+9 setelah normalisasi) sehingga
  dibuang (label kontradiktif). Kasus **tanpa nama penulis praktis tidak ada** di dataset ini.
- Setelah pembersihan: 11.534 baris (5.995 valid, 5.539 halu). Split 80/20 dan CV 5-fold **per grup**: referensi asli
  dan semua manipulasinya selalu berada di sisi yang sama, jadi model diuji pada makalah yang belum pernah dilihatnya.

| Model | CV ROC-AUC | Uji ROC-AUC | Parameter terbaik |
|---|---|---|---|
| Logistic Regression | 0,618 ± 0,008 | 0,629 | C = 0,1 |
| Random Forest | 0,549 ± 0,007 | 0,548 | min_samples_leaf = 5 |
| XGBoost | 0,570 ± 0,012 | 0,572 | max_depth = 3, 600 pohon |

Model terpilih (CV): Logistic Regression, probabilitas dikalibrasi (Platt, bobot kelas seimbang). Pada data uji:
ROC-AUC 0,63 · F1 halu 0,58 · balanced accuracy 0,58. Ablasi: judul saja 0,61; author saja 0,57; keduanya 0,63.

Model ML saja, per jenis manipulasi (data uji, ambang 0,5):

| Jenis | n | Terdeteksi halu | CS | non-CS |
|---|---|---|---|---|
| Kata judul diacak | 294 | 74% | 68% | 81% |
| Kata judul dihapus | 289 | 55% | 56% | 55% |
| Susunan nama penulis dibalik | 262 | 60% | 55% | 64% |
| Nama penulis tidak lengkap / satu nama | 264 | 48% | 45% | 52% |
| *Referensi asli yang salah dituduh halu* | 1.198 | *43%* | 41% | 45% |

**Cara membacanya:** setiap referensi halu hanya berbeda sedikit dari referensi asli (urutan kata, satu-dua kata,
nama). Embedding kalimat MultiBERT hampir tidak berubah oleh perubahan sekecil itu, sehingga model murni-teks hanya
sedikit lebih baik dari menebak; yang paling terdeteksi adalah judul yang diacak. Manipulasi semacam ini ditangkap oleh
**pemeriksaan Tahap 2** terhadap rekaman asli (lihat "Uji manipulasi dataset" di bawah), dan keputusan akhir sebagian
besar bertumpu pada bukti tersebut. Skor di atas mengukur dataset sintetis ini, bukan halusinasi LLM di dunia nyata.

**Filter domain** (Tahap 4) kini dilatih dengan referensi CS **dan non-CS** dari `dataset_scope_6000.csv` (judul halu
ikut dipakai dengan scope sumbernya): Logistic Regression atas embedding judul → P(Ilmu Komputer). Ambang 0,43 dipilih
agar ≤ 5% judul CS salah diisolasi (out-of-fold). Data uji: AUC 0,989; CS salah diisolasi 3,6%; non-CS terisolasi
92,5%. Daftar tulisan tangan kecil: 100% non-CS terisolasi, 0% CS salah terisolasi. Referensi yang ditemukan di sumber
CS (dataset seed, DBLP, arXiv `cs.*`, OpenAlex "Computer Science") otomatis dianggap in-scope.

**Referensi di luar scope tetap dinilai.** Statusnya DI LUAR SCOPE, tetapi kartunya juga menampilkan VALID / TIDAK VALID
dan perkiraan % halu, ditambah peringatan agar dicek ulang secara manual. Pie chart "Persentase Keseluruhan" membagi
semua referensi: hijau = Valid, merah = Halu, kuning = Luar Scope (+ abu-abu = Tidak Terbaca bila ada), dengan nama dan
persen di dalam irisan; di bawahnya rincian luar scope (berapa valid / tidak valid).

## Uji manipulasi dataset (end-to-end)

`python evaluate_system.py --set manipulasi` menguji pipeline utuh pada 140 referensi APA: 40 asli (20 CS + 20 non-CS
dari `dataset_scope_6000.csv`), 80 manipulasi dari `dataset_halu_6000_2kolom.csv` (20 per jenis), dan 20 referensi asli
yang penulisnya dihapus (kasus "tanpa penulis" yang tidak ada di dataset halu). Laporan: `reports/manipulation_evaluation.md`.

| Kelompok | n | Pipeline lengkap benar | Model ML saja benar |
|---|---|---|---|
| Asli, Ilmu Komputer | 20 | **100%** | 65% |
| Asli, di luar Ilmu Komputer | 20 | **100%** | 50% |
| Halu: kata judul diacak | 20 | **100%** | 95% |
| Halu: kata judul dihapus | 20 | **100%** | 65% |
| Halu: susunan nama penulis dibalik | 20 | **90%** | 70% |
| Halu: penulis tidak lengkap / satu nama | 20 | **65%** | 40% |
| Halu: tanpa nama penulis | 20 | **95%** | 100% |

Recall halusinasi **90%**, precision **100%** (tidak ada referensi asli yang dituduh), F1 0,95. Yang lolos hampir semuanya
"penulis tidak lengkap": setelah diformat APA, nama yang dipotong sering tetap menjadi sitasi yang benar (mis.
"Oktavian, D., & Putra, B." untuk Dipo Oktavian dan Bayu Anugrah Putra), sehingga memang tidak bisa dibedakan dari sitasi
asli. Model ML final dilatih pada seluruh dataset, jadi kolom "ML saja" optimistis.

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
- **Pencocokan dengan rekaman asli** (Tahap 2) sengaja ketat terhadap pola manipulasi tetapi toleran terhadap variasi
  penulisan yang wajar: subjudul (setelah `:`) boleh tidak ditulis, kata sandang di awal dan `&`/`and` dianggap sama,
  nama tengah boleh disingkat atau dihilangkan, dan daftar penulis boleh dipersingkat **asal ada "et al." / "dkk."**.
  Yang ditandai: kata judul diacak, kata judul dihilangkan, penulis tidak dicantumkan sama sekali, nama depan ditulis
  sebagai nama belakang, hanya nama depan yang ditulis, dan penulis dihilangkan tanpa "et al.". Judul yang terpotong
  oleh GROBID (judul lengkapnya tertulis utuh di teks) tidak dihitung sebagai manipulasi. Bila GROBID tidak menemukan
  judul (mis. APA tanpa penulis "Judul. (2020). Sumber."), judul diambil dari teks mentah, sehingga yang diperiksa
  tetap judul yang ditulis pengguna, bukan metadata DOI-nya.
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
