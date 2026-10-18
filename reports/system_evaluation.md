# Evaluasi sistem CIVA (end-to-end)

196 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML, filter domain) dalam 4.9 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | DI LUAR SCOPE | TIDAK TERBACA |
|---|---|---|---|---|---|---|---|
| Palsu: DOI karangan | 24 | 96% | 54% | 0 | 23 | 1 | 0 |
| Palsu: karangan penuh | 24 | 83% | 92% | 0 | 20 | 4 | 0 |
| Palsu: judul nyata + penulis salah | 24 | 96% | 38% | 0 | 23 | 1 | 0 |
| Palsu: tahun digeser >= 3 | 24 | 96% | 33% | 1 | 23 | 0 | 0 |
| Nyata, ada di dataset (lokal) | 20 | 100% | 60% | 20 | 0 | 0 | 0 |
| Nyata, tidak ada di dataset (API eksternal) | 80 | 82% | 56% | 66 | 1 | 11 | 2 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **93%** (89 dari 96 referensi palsu terdeteksi)
- Precision: **99%** (1 referensi nyata salah dituduh halu)
- F1: **0.96** | akurasi keseluruhan: 96%
- Referensi nyata: salah dituduh HALU 1%, salah dinyatakan DI LUAR SCOPE 11%, tidak terbaca 2%
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 54%, alarm palsu 41%

## Catatan metodologi

- Referensi nyata diambil dari OpenAlex; sistem mencari ke Crossref/OpenAlex juga, sehingga ada unsur sirkular: angka 'nyata' mengukur kebenaran parsing + pencocokan + keputusan, bukan cakupan indeks.
- Referensi palsu bersifat sintetis (karangan tangan dan korupsi terkontrol: penulis, tahun, DOI). Halusinasi LLM nyata bisa lebih beragam.
- Referensi yang tidak berDOI atau tidak terindeks (buku, skripsi, jurnal lokal) kurang terwakili.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/eval_set.json (reproduksibel).
