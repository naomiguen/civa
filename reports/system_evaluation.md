# Evaluasi sistem CIVA (end-to-end)

196 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML, filter domain) dalam 4.0 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | DI LUAR SCOPE | TIDAK TERBACA |
|---|---|---|---|---|---|---|---|
| Palsu: DOI karangan | 24 | 100% | 21% | 0 | 22 | 2 | 0 |
| Palsu: karangan penuh | 24 | 100% | 17% | 0 | 21 | 3 | 0 |
| Palsu: judul nyata + penulis salah | 24 | 100% | 29% | 0 | 22 | 2 | 0 |
| Palsu: tahun digeser >= 3 | 24 | 100% | 17% | 0 | 24 | 0 | 0 |
| Nyata, ada di dataset (lokal) | 20 | 95% | 60% | 19 | 1 | 0 | 0 |
| Nyata, tidak ada di dataset (API eksternal) | 80 | 90% | 75% | 61 | 8 | 11 | 0 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **100%** (96 dari 96 referensi palsu terdeteksi)
- Precision: **91%** (9 referensi nyata salah dituduh halu)
- F1: **0.96** | akurasi keseluruhan: 95%
- Referensi nyata: salah dituduh HALU 9%, dinyatakan DI LUAR SCOPE 11%, tidak terbaca 0%
- Referensi DI LUAR SCOPE tetap dinilai valid/halu (dengan peringatan); kolom Benar dan angka deteksi memakai penilaian itu.
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 21%, alarm palsu 28%

## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok

- Palsu: DOI karangan: DOI (24)
- Palsu: karangan penuh: DOI (11)
- Palsu: judul nyata + penulis salah: penulis (22)
- Palsu: tahun digeser >= 3: tahun (24), jumlah penulis (1)
- Nyata, ada di dataset (lokal): jumlah penulis (1)
- Nyata, tidak ada di dataset (API eksternal): jumlah penulis (5), DOI (2)

## Catatan metodologi

- Referensi nyata diambil dari OpenAlex; sistem mencari ke Crossref/OpenAlex juga, sehingga ada unsur sirkular: angka 'nyata' mengukur kebenaran parsing + pencocokan + keputusan, bukan cakupan indeks.
- Referensi palsu bersifat sintetis (karangan tangan dan korupsi terkontrol: penulis, tahun, DOI). Halusinasi LLM nyata bisa lebih beragam.
- Referensi yang tidak berDOI atau tidak terindeks (buku, skripsi, jurnal lokal) kurang terwakili.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/eval_set.json (reproduksibel).
