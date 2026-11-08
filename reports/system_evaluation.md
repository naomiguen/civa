# Evaluasi sistem CIVA (end-to-end)

196 referensi berlabel, diperiksa pipeline lengkap (GROBID asli, API eksternal, model ML) dalam 3.3 menit.

| Kelompok | n | Benar | ML saja benar | VALID | HALU | TIDAK TERBACA |
|---|---|---|---|---|---|---|
| Palsu: DOI karangan | 24 | 100% | 38% | 0 | 24 | 0 |
| Palsu: karangan penuh | 24 | 100% | 29% | 0 | 24 | 0 |
| Palsu: judul nyata + penulis salah | 24 | 96% | 21% | 1 | 23 | 0 |
| Palsu: tahun digeser >= 3 | 24 | 100% | 38% | 0 | 24 | 0 |
| Nyata, ada di dataset (lokal) | 20 | 100% | 75% | 20 | 0 | 0 |
| Nyata, tidak ada di dataset (API eksternal) | 80 | 98% | 69% | 78 | 2 | 0 |

## Deteksi halusinasi (positif = referensi palsu)

- Recall: **99%** (95 dari 96 referensi palsu terdeteksi)
- Precision: **98%** (2 referensi nyata salah dituduh halu)
- F1: **0.98** | akurasi keseluruhan: 98%
- Referensi nyata: salah dituduh HALU 2%, tidak terbaca 0%
- Pembanding, model ML saja (tanpa bukti pangkalan data): recall 31%, alarm palsu 30%

## Pola yang ditemukan Tahap 2 (kolom mismatch) per kelompok

- Palsu: DOI karangan: DOI (24), penulis (1)
- Palsu: karangan penuh: DOI (10)
- Palsu: judul nyata + penulis salah: penulis (19), DOI (1)
- Palsu: tahun digeser >= 3: tahun (21)
- Nyata, ada di dataset (lokal): -
- Nyata, tidak ada di dataset (API eksternal): DOI (1)

## Catatan metodologi

- Referensi nyata diambil dari OpenAlex; sistem mencari ke Crossref/OpenAlex juga, sehingga ada unsur sirkular: angka 'nyata' mengukur kebenaran parsing + pencocokan + keputusan, bukan cakupan indeks.
- Referensi palsu bersifat sintetis (karangan tangan dan korupsi terkontrol: penulis, tahun, DOI). Halusinasi LLM nyata bisa lebih beragam.
- Referensi yang tidak berDOI atau tidak terindeks (buku, skripsi, jurnal lokal) kurang terwakili.
- GROBID: http://localhost:8070. Set evaluasi tersimpan di reports/eval_set.json (reproduksibel).
