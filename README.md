# CIVA - Citation Intelligence & Verification Assistant

Sistem verifikasi daftar pustaka berbasis Machine Learning untuk mendeteksi sitasi halusinasi pada literatur
ilmiah dari **semua bidang ilmu** (tanpa filter domain). Frontend **Streamlit**, backend **FastAPI**, ekstraksi
**GROBID** (Docker), klasifikasi **MultiBERT + XGBoost / Random Forest / Logistic Regression**.

## Alur sistem

```
teks daftar pustaka
   │
   ├─ TAHAP 1  GROBID  POST /api/processCitation  → <title>, <author> (+ tahun, DOI, jenis)
   ├─ TAHAP 2  Pemeriksaan DOI langsung (Crossref /works/{doi}, registri doi.org) bila DOI dicantumkan, lalu
   │            Retrieval + caching
   │            DB lokal (SQLite) → Crossref → arXiv → OpenAlex → OpenLibrary (buku) → DBLP
   │            (berurutan, berhenti saat terverifikasi)
   │            temuan dari API eksternal di-INSERT ke DB lokal (berikutnya cukup dari lokal)
   │            bila ditemukan, dicocokkan dgn rekaman asli: kata judul diacak / dihilangkan, penulis berbeda /
   │            tidak dicantumkan / susunan nama terbalik / urutan diacak / sebagian dihilangkan, tahun, DOI → "mismatch"
   ├─ TAHAP 3  ML MURNI TEKS: embedding MultiBERT judul (384) + author (384) = 768-d → classifier → P(valid)
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

**3. Latih model** (sekali; membaca `dataset_v2\dataset_final.csv`)

```powershell
python train_model.py            # grid penuh, ±10 menit di CPU
python train_model.py --quick    # grid kecil, ±1 menit bila cache embedding sudah ada
```

Menghasilkan di `models\`: `civa_pipeline.joblib` (pipeline MultiBERT → classifier terkalibrasi),
`multibert_tokenizer.joblib`, `training_report.json`, `model_comparison.csv`.
Model MultiBERT (±470 MB) diunduh otomatis dari HuggingFace pada pemakaian pertama.

**4. Jalankan backend** (terminal baru, venv aktif)

```powershell
uvicorn backend.main:app --port 8000
```

Saat pertama kali jalan, database lokal `data\civa.db` terisi otomatis dari referensi asli `dataset_final.csv`
(±11.900 referensi: 6.000 jurnal dari 26 bidang ilmu + buku). Bila isi dataset seed berubah, baris seed diisi ulang
otomatis saat backend start; rekaman hasil API eksternal tetap disimpan.
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
├─ civa_core.py          preprocessing + MultiBERTEmbedder + classifier terkalibrasi (dipakai training DAN backend)
├─ evaluate_system.py    evaluasi end-to-end (set umum & uji manipulasi dataset)
├─ backend/
│  ├─ main.py            FastAPI: alur Tahap 1-3 + keputusan
│  ├─ grobid_client.py   klien GROBID + parser TEI (abaikan namespace)
│  ├─ retrieval.py       DB lokal → Crossref → arXiv → OpenAlex → OpenLibrary → DBLP, pencocokan metadata
│  ├─ database.py        SQLite: knowledge base + cache hasil API + cache "tidak ditemukan"
│  ├─ verdict.py         penggabungan P(valid) + bukti, teks "Alasan"
│  ├─ textutil.py        kemiripan judul, kecocokan penulis, pemecah referensi
│  ├─ schemas.py · config.py
├─ frontend/app.py       Streamlit (mengikuti desain UI/UX) · frontend/run.py peluncurnya
├─ tests/test_core.py
├─ dataset_v2/           dataset_final.csv (latih/uji + seed DB, jurnal + buku) · dataset_halu.csv (subset halu)
├─ dataset/              versi lama (tanpa buku), tidak dipakai lagi
└─ models/ · data/       hasil pelatihan · database lokal (dibuat otomatis)
```

## API

`POST /api/verify` dengan `{"text": "<daftar pustaka, maks. 10 referensi>"}` mengembalikan:

```jsonc
{
  "summary": {"total": 5, "valid": 3, "halu": 2, "unparsed": 0, "valid_percent": 60.0},
  "results": [{
    "title": "...", "authors_display": "Zhang, Y. & Chen, L.", "year": 2023, "type": "Jurnal",
    "status": "valid|halu|unparsed", "label": "VALID", "confidence": 0.95, "reason": "Alasan: ...",
    "halu_probability": 0.05,       // perkiraan peluang halusinasi (referensi yang terbaca)
    "ml":       {"label": "valid", "probability_valid": 0.72, "probability_halu": 0.28},   // murni teks
    "evidence": {"status": "verified|mismatch|not_found", "source": "local|crossref|arxiv|openalex|dblp", ...}
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
| `CIVA_NOT_FOUND_WEIGHT` | `-2.0` | pergeseran logit bila tidak ditemukan di mana pun |
| `CIVA_TITLE_MATCH_THRESHOLD` | `0.90` | kemiripan judul minimum agar dianggap ditemukan |
| `CIVA_MISMATCH_TITLE_THRESHOLD` | `0.97` | kemiripan judul minimum untuk menuduh "penulis/tahun tidak sesuai" |
| `CIVA_DOI_CONFLICT_TITLE_THRESHOLD` | `0.60` | DOI sah milik makalah lain hanya dituduh bila judul jelas berbeda |
| `CIVA_GROBID_TIMEOUT` | `90` | detik menunggu jawaban GROBID per percobaan (diulang 3x) |
| `CIVA_MAX_REFERENCES` | `10` | batas referensi per pemeriksaan |

Set `CIVA_EVIDENCE_WEIGHT=0` dan `CIVA_NOT_FOUND_WEIGHT=0` agar verdict **sepenuhnya** dari model ML.

## Hasil evaluasi (apa adanya)

**Data** (`dataset_v2/dataset_final.csv`): 19.584 referensi.
- 11.894 **asli**: 6.000 artikel jurnal dari 26 bidang ilmu (± 230 per bidang, termasuk Computer Science; Inggris 70%,
  Indonesia 30%) dan 5.894 **buku** dari OpenLibrary (± separuh Ilmu Komputer, sebagian besar berbahasa Inggris; kolom
  `jenis_referensi = book`, `link_referensi` = URL OpenLibrary).
- 7.690 **halu** hasil perusakan referensi asli: kata judul diacak 2.000, kata judul dihapus 2.000, buku dengan
  judul/penulis dimanipulasi (`mismatch_buku`) 1.713, penulis tinggal satu nama 498, tanpa nama penulis 496, urutan
  penulis diacak 495, susunan nama penulis dibalik 488.

Kolom `grup_id` mengikat referensi asli dengan semua perusakannya, dan kolom `split` (latih 80% / uji 20%) sudah
ditetapkan per grup, jadi model diuji pada referensi yang belum pernah dilihatnya. Pembersihan membuang 214 baris (halu
yang identik dengan aslinya setelah normalisasi, duplikat, dan judul/penulis tak terpakai), sehingga yang dipakai
19.370 baris (latih 15.502 / uji 3.868). `dataset_halu.csv` berisi baris halu yang sama (subset), hanya sebagai rujukan.
Folder `dataset/` adalah versi lama tanpa buku (11.977 baris) dan tidak dipakai lagi.

| Model | CV ROC-AUC (latih, 5-fold) | Uji ROC-AUC | Parameter terbaik |
|---|---|---|---|
| Logistic Regression | 0,721 ± 0,006 | 0,719 | C = 0,03 |
| Random Forest | 0,652 ± 0,003 | 0,649 | min_samples_leaf = 5 |
| XGBoost | 0,686 ± 0,006 | 0,680 | max_depth = 5, 600 pohon |

Model terpilih (CV): Logistic Regression, probabilitas dikalibrasi (Platt, bobot kelas seimbang), **dilatih hanya pada
split latih** sehingga split uji tidak pernah dilihat model yang dipakai backend. Dilatih dengan grid hiperparameter
lengkap (`python train_model.py`, ±22 menit di CPU). Pada data uji: ROC-AUC 0,72 · F1 halu 0,60 · balanced accuracy
0,66 · MCC 0,31.
Ablasi: judul saja 0,71; author saja 0,60; keduanya 0,72. AUC per bahasa: jurnal Inggris 0,71, Indonesia 0,67, buku
(kode bahasa `eng`) 0,62.

Model ML saja, per jenis (data uji, ambang 0,5):

| Jenis | n | Terdeteksi halu |
|---|---|---|
| Kata judul diacak | 402 | 82% |
| Kata judul dihapus | 431 | 72% |
| Buku: judul/penulis dimanipulasi | 325 | 35% |
| Penulis tinggal satu nama | 95 | 61% |
| Tanpa nama penulis | 91 | 100% |
| Urutan penulis diacak | 100 | 48% |
| Susunan nama penulis dibalik | 78 | 59% |
| *Referensi asli yang salah dituduh halu* | 2.346 | *34%* |

**Cara membacanya:** referensi halu hanya berbeda sedikit dari aslinya (urutan kata, satu-dua kata, nama), dan embedding
kalimat MultiBERT hampir tidak berubah oleh perubahan sekecil itu, sehingga model murni-teks hanya memisahkan secara
sedang. "Tanpa nama penulis" 100% karena di dataset hanya referensi halu yang tidak punya penulis. Manipulasi buku
hampir tidak terlihat oleh model (35%, setara dengan alarm palsu pada referensi asli), karena perubahannya sangat kecil
(nama penulis dibalik/diacak, inisial dipindah). Manipulasi semacam ini ditangkap oleh **pemeriksaan Tahap 2** terhadap
rekaman asli (lihat "Uji manipulasi dataset" di bawah), dan keputusan akhir sebagian besar bertumpu pada bukti tersebut.
Skor di atas mengukur dataset sintetis ini, bukan halusinasi LLM di dunia nyata.

**Semua bidang ilmu diperlakukan sama.** Tidak ada lagi filter domain / status "di luar scope": referensi kedokteran,
hukum, kimia, dan seterusnya diperiksa dengan cara yang sama seperti referensi Ilmu Komputer. Pie chart "Persentase
Keseluruhan" membagi semua referensi menjadi Valid (hijau), Halu (merah), dan Tidak Terbaca (abu-abu, bila ada).

## Uji manipulasi dataset (end-to-end)

`python evaluate_system.py --set manipulasi` menguji pipeline utuh (GROBID asli, API eksternal, model ML) pada 220
referensi APA dari **split uji** `dataset_v2/dataset_final.csv` (tidak pernah dilihat model): 40 artikel jurnal asli
dari berbagai bidang, 40 buku asli, dan 20 per jenis perusakan. Jurnal ditulis dengan nama jurnal, buku tanpa nama
penerbit (tidak ada di dataset), semuanya tanpa DOI. Laporan: `reports/manipulation_evaluation.md` (11 November 2026).

| Kelompok | n | Pipeline lengkap benar | Model ML saja benar |
|---|---|---|---|
| Asli: jurnal (semua bidang) | 40 | **100%** | 45% |
| Asli: buku | 40 | **92%** | 85% |
| Halu: kata judul diacak | 20 | **100%** | 75% |
| Halu: kata judul dihapus | 20 | **100%** | 70% |
| Halu: buku (judul/penulis dimanipulasi) | 20 | **90%** | 35% |
| Halu: penulis tinggal satu nama | 20 | **100%** | 60% |
| Halu: tanpa nama penulis | 20 | **95%** | 85% |
| Halu: urutan penulis diacak | 20 | **95%** | 45% |
| Halu: susunan nama penulis dibalik | 20 | **95%** | 45% |

Recall halusinasi **96%** (135 dari 140), precision **98%**, F1 0,97. Pembanding model ML saja: recall 59%, alarm
palsu 35%. Kolom "ML saja" jujur karena model tidak pernah melihat split uji; selisihnya dengan pipeline lengkap adalah
sumbangan pemeriksaan Tahap 2.

Tiga buku asli yang dituduh halu punya data penulis yang tidak lazim di dataset OpenLibrary: lembaga atau penerbit
yang tercatat sebagai penulis ("Research and Education Association", "Lippincott Williams & Wilkins") dan keterangan
yang ikut di nama ("Zhixian Yan (Computer scientist)"). Dua buku halu yang lolos hanya berbeda tipis dari aslinya
pada nama penulis tunggal atau nama lembaga (mis. "S. A. Inc"). Pada buku, **tahun tidak
dinilai** (edisi dan cetak ulang wajar berbeda tahun) dan buku lain yang kebetulan berjudul sama dengan penulis yang
sama sekali berbeda tidak dianggap bukti manipulasi.

## Evaluasi sistem end-to-end

`python evaluate_system.py` mengukur pipeline utuh pada 196 referensi berlabel, dengan backend & database sementara
(tidak menyentuh `data/civa.db`; GROBID harus berjalan). Set: 100 referensi nyata (80 dari OpenAlex, **semua bidang**,
tidak ada di dataset; 20 dari dataset) dan 96 palsu (DOI karangan, karangan penuh, judul nyata + penulis makalah lain,
tahun digeser ≥ 3). Laporan: `reports/system_evaluation.md`, rincian: `reports/system_eval_results.csv`
(12 November 2026, model dan database `dataset_v2`).

| Kelompok | n | Pipeline lengkap benar | Model ML saja benar |
|---|---|---|---|
| Nyata, tidak ada di dataset (API eksternal) | 80 | **99%** | 50% |
| Nyata, ada di dataset (lokal) | 20 | **100%** | 45% |
| Palsu: DOI karangan | 24 | **100%** | 62% |
| Palsu: karangan penuh | 24 | **100%** | 50% |
| Palsu: judul nyata + penulis salah | 24 | **96%** | 42% |
| Palsu: tahun digeser ≥ 3 | 24 | **100%** | 46% |

Recall halusinasi **99%** (95 dari 96), precision **99%**, F1 0,99. Pembanding model ML saja: recall 50%, alarm
palsu 51%. Satu-satunya referensi nyata yang dituduh mencantumkan DOI yang memang tidak terdaftar (404 di Crossref
maupun registri doi.org). Satu referensi palsu lolos karena nama penulisnya beraksara Sirilik sehingga tidak bisa
dibandingkan. Set uji bersifat sintetis dan sirkular untuk cakupan indeks (referensi nyata
diambil dari OpenAlex, yang juga dipakai sistem; lihat catatan metodologi di laporan), jadi jangan dibaca sebagai
akurasi di dunia nyata.

## Catatan

- **Pencarian eksternal** memakai API publik. arXiv dibatasi 1 request / 3 detik, jadi pemeriksaan 10 referensi
  yang tidak ditemukan bisa memakan puluhan detik pada kali pertama; hasil yang ditemukan dan pencarian yang
  gagal di-cache sehingga pemeriksaan ulang hampir seketika. "Tidak ditemukan" hanya di-cache bila semua sumber
  menjawab; sumber yang gagal 3 kali beruntun dilewati 5 menit. Per November 2026 **DBLP** membalas dengan halaman
  cek bot (bukan JSON), jadi praktis dilewati dan pencarian memakai Crossref, arXiv, OpenAlex, dan OpenLibrary.
- **Buku** dicari juga di **OpenLibrary** (tanpa API key; entri tanpa nama penulis diabaikan). Buku yang ditulis
  dengan edisinya ("3rd ed.", "Eighth Edition", "Edisi ke-2") tidak dituduh karena tahunnya berbeda dari edisi lain.
  Buku asli yang judulnya generik dan tidak ada di sumber mana pun, tetapi judulnya sama persis dengan artikel lain di
  Crossref, masih bisa dituduh "penulis tidak sesuai".
- **Penulis tambahan**: bila semua penulis asli tercantum tetapi referensi memuat lebih banyak nama daripada karya
  aslinya (ada nama karangan yang ditambahkan), referensi ditandai.
- **Pencocokan dengan rekaman asli** (Tahap 2) sengaja ketat terhadap pola manipulasi tetapi toleran terhadap variasi
  penulisan yang wajar: subjudul (setelah `:`) boleh tidak ditulis, kata sandang di awal dan `&`/`and` dianggap sama,
  nama tengah boleh disingkat atau dihilangkan, dan daftar ≥ 2 penulis boleh dipersingkat. Yang ditandai: kata judul
  diacak, kata judul dihilangkan, penulis tidak dicantumkan sama sekali, nama depan ditulis sebagai nama belakang, hanya
  nama depan yang ditulis, urutan penulis diacak, dan hanya SATU penulis ditulis padahal aslinya lebih banyak (tanpa
  "et al." / "dkk."). Nama penulis yang tertulis tetapi tidak terbaca (mis. aksara Sirilik/Tionghoa) tidak dianggap
  "tanpa penulis"; perbandingan penulisnya dilewati. Judul yang terpotong
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
