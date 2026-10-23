# Uji manipulasi dataset CIVA (end-to-end)

140 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML, filter domain) dalam 5.5 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | DI LUAR SCOPE | TIDAK TERBACA |
|---|---|---|---|---|---|---|---|
| Asli, Ilmu Komputer | 20 | 100% | 65% | 20 | 0 | 0 | 0 |
| Asli, di luar Ilmu Komputer | 20 | 100% | 50% | 3 | 0 | 17 | 0 |
| Halu: kata judul diacak | 20 | 100% | 95% | 0 | 10 | 10 | 0 |
| Halu: kata judul dihapus | 20 | 100% | 65% | 0 | 10 | 10 | 0 |
| Halu: susunan nama penulis dibalik | 20 | 90% | 70% | 1 | 9 | 10 | 0 |
| Halu: penulis tidak lengkap / satu nama | 20 | 65% | 40% | 4 | 8 | 8 | 0 |
| Halu: tanpa nama penulis (disusun dari data asli) | 20 | 95% | 100% | 1 | 9 | 10 | 0 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **90%** (90 dari 100 referensi palsu terdeteksi)
- Precision: **100%** (0 referensi nyata salah dituduh halu)
- F1: **0.95** | akurasi keseluruhan: 93%
- Referensi nyata: salah dituduh HALU 0%, dinyatakan DI LUAR SCOPE 42%, tidak terbaca 0%
- Referensi DI LUAR SCOPE tetap dinilai valid/halu (dengan peringatan); kolom Benar dan angka deteksi memakai penilaian itu.
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 74%, alarm palsu 42%

## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok

- Asli, Ilmu Komputer: -
- Asli, di luar Ilmu Komputer: -
- Halu: kata judul diacak: urutan judul (20), tahun (1), tanpa penulis (1), jumlah penulis (1)
- Halu: kata judul dihapus: kata judul (6), jumlah penulis (1)
- Halu: susunan nama penulis dibalik: penulis (9), jumlah penulis (5), susunan nama (3)
- Halu: penulis tidak lengkap / satu nama: jumlah penulis (8), nama penulis (4), susunan nama (1)
- Halu: tanpa nama penulis (disusun dari data asli): tanpa penulis (16), kata judul (1)

## Catatan metodologi

- Referensi asli dari dataset_scope_6000.csv; referensi halu dari dataset_halu_6000_2kolom.csv, masing-masing ditautkan ke referensi asli sumbernya (tahun & DOI diambil dari sumber; 50% diberi DOI). Diformat APA.
- 'Tanpa nama penulis' disusun dari referensi asli yang penulisnya dihapus, karena dataset halu praktis tidak memuat kasus ini.
- Model ML final dilatih pada SELURUH dataset, termasuk baris-baris ini: kolom 'ML saja benar' optimistis.
- Semua referensi halu di sini adalah manipulasi makalah NYATA, sehingga makalah aslinya bisa ditemukan di basis data lokal/API; yang diuji adalah pemeriksaan kecocokan Tahap 2 terhadap rekaman asli tersebut.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/manipulation_eval_set.json (reproduksibel).
