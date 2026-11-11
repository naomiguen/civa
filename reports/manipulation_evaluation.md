# Uji manipulasi dataset CIVA (end-to-end)

220 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML) dalam 7.3 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | TIDAK TERBACA |
|---|---|---|---|---|---|---|
| Halu: kata judul diacak | 20 | 100% | 75% | 0 | 20 | 0 |
| Asli: jurnal (semua bidang) | 40 | 100% | 45% | 40 | 0 | 0 |
| Asli: buku | 40 | 92% | 85% | 37 | 3 | 0 |
| Halu: kata judul dihapus | 20 | 100% | 70% | 0 | 20 | 0 |
| Halu: buku (judul/penulis dimanipulasi) | 20 | 90% | 35% | 2 | 18 | 0 |
| Halu: susunan nama penulis dibalik | 20 | 95% | 45% | 1 | 19 | 0 |
| Halu: tanpa nama penulis | 20 | 95% | 85% | 1 | 19 | 0 |
| Halu: penulis tinggal satu nama | 20 | 100% | 60% | 0 | 20 | 0 |
| Halu: urutan penulis diacak | 20 | 95% | 45% | 1 | 19 | 0 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **96%** (135 dari 140 referensi palsu terdeteksi)
- Precision: **98%** (3 referensi nyata salah dituduh halu)
- F1: **0.97** | akurasi keseluruhan: 96%
- Referensi nyata: salah dituduh HALU 4%, tidak terbaca 0%
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 59%, alarm palsu 35%

## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok

- Halu: kata judul diacak: urutan judul (19), kata judul (1)
- Asli: jurnal (semua bidang): -
- Asli: buku: jumlah penulis (2)
- Halu: kata judul dihapus: kata judul (2)
- Halu: buku (judul/penulis dimanipulasi): urutan judul (8), susunan nama (5), urutan penulis (4), kata judul (1), penulis (1)
- Halu: susunan nama penulis dibalik: susunan nama (19)
- Halu: tanpa nama penulis: tanpa penulis (16)
- Halu: penulis tinggal satu nama: jumlah penulis (20)
- Halu: urutan penulis diacak: urutan penulis (16), penulis (1), jumlah penulis (1)

## Catatan metodologi

- Semua referensi diambil dari split UJI dataset_final.csv (asli + perusakannya), diformat APA tanpa DOI.
- Model ML final dilatih HANYA pada split latih, jadi kolom 'ML saja benar' jujur (data belum pernah dilihat).
- Semua referensi halu di sini adalah manipulasi makalah NYATA, sehingga makalah aslinya bisa ditemukan di basis data lokal/API; yang diuji adalah pemeriksaan kecocokan Tahap 2 terhadap rekaman asli tersebut.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/manipulation_eval_set.json (reproduksibel).
