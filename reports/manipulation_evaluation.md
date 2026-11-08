# Uji manipulasi dataset CIVA (end-to-end)

160 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML) dalam 6.8 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | TIDAK TERBACA |
|---|---|---|---|---|---|---|
| Halu: kata judul diacak | 20 | 100% | 55% | 0 | 20 | 0 |
| Asli (semua bidang) | 40 | 100% | 57% | 40 | 0 | 0 |
| Halu: kata judul dihapus | 20 | 100% | 65% | 0 | 20 | 0 |
| Halu: susunan nama penulis dibalik | 20 | 90% | 50% | 2 | 18 | 0 |
| Halu: tanpa nama penulis | 20 | 100% | 95% | 0 | 20 | 0 |
| Halu: penulis tinggal satu nama | 20 | 95% | 60% | 1 | 19 | 0 |
| Halu: urutan penulis diacak | 20 | 95% | 10% | 1 | 19 | 0 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **97%** (116 dari 120 referensi palsu terdeteksi)
- Precision: **100%** (0 referensi nyata salah dituduh halu)
- F1: **0.98** | akurasi keseluruhan: 98%
- Referensi nyata: salah dituduh HALU 0%, tidak terbaca 0%
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 56%, alarm palsu 42%

## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok

- Halu: kata judul diacak: urutan judul (17), kata judul (2)
- Asli (semua bidang): -
- Halu: kata judul dihapus: kata judul (3), penulis (1), tahun (1)
- Halu: susunan nama penulis dibalik: susunan nama (16), tanpa penulis (2)
- Halu: tanpa nama penulis: tanpa penulis (15), tahun (1)
- Halu: penulis tinggal satu nama: jumlah penulis (17), penulis (1)
- Halu: urutan penulis diacak: urutan penulis (18), penulis (1)

## Catatan metodologi

- Semua referensi diambil dari split UJI dataset_final.csv (asli + perusakannya), diformat APA tanpa DOI.
- Model ML final dilatih HANYA pada split latih, jadi kolom 'ML saja benar' jujur (data belum pernah dilihat).
- Semua referensi halu di sini adalah manipulasi makalah NYATA, sehingga makalah aslinya bisa ditemukan di basis data lokal/API; yang diuji adalah pemeriksaan kecocokan Tahap 2 terhadap rekaman asli tersebut.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/manipulation_eval_set.json (reproduksibel).
