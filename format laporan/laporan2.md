# Template Laporan Praktikum (versi ringkas / checklist)

Template cepat untuk mengisi laporan. Untuk versi lengkap & contoh terisi, lihat `format laporan/laporan1.md`.
Cara pakai: salin skeleton ini ke `laporan/laporan{N}{metode}.md`, lalu ganti `{placeholder}`.

---

## Skeleton

```markdown
# Laporan {N} — {Judul Metode}

Notebook: `notebook/{nama}.ipynb`

## Dataset
- Sumber Kaggle (`kagglehub.dataset_download("{owner}/{slug}")`): {nama}, {ukuran}, **{n} baris**.
- Fitur: {daftar}; Target: {target}.
- Distribusi kelas: {kelas} = {n} ({p}%), ...

## Rumus
- {Rumus 1 (entropy/prior/...)}
- {Rumus 2 (gain/likelihood/...)}
- {Rumus 3 (posterior/prediksi)}
- Akurasi = benar / n
- RMSE = sqrt(1/n Σ(y - ŷ)²)

## 1. Pengujian + akurasi/RMSE
- Split 80/20 seed 42 → latih {n}, uji {n}.
- Akurasi: **{}** ; RMSE: **{}** ; Baseline: {}%.
- Confusion: TP {}, FN {}, FP {}, TN {}.

## 2. Data lebih banyak (min 100)
- Dari {n_lama} → **{n_baru} baris** via Kaggle API.

## 3. Perhitungan manual {n} data
- Tabel {n} data + label.
- {Rumus 1} manual = {}.
- {Rumus 2} manual per fitur/kelas (tabel).
- {Rumus 3} posterior contoh data #{} → prediksi (benar/salah).
- Resubstitution: akurasi {}%, RMSE {} (salah di #{}, #{}).

## 4. Perbandingan metode lain ({n} data sama)
- {Metode_2}: akurasi {}%, RMSE {}.
- Tabel perbandingan + data salah masing-masing.

## 5. Perbandingan besar (data uji {n})
- {Metode_1} vs {Metode_2}: akurasi {} / {}; RMSE {} / {}.
- {Mengapa identik/berbeda.}

## 6. Mana lebih baik?
- {Metode_1} jika: ... ; {Metode_2} jika: ...
- Kesimpulan: ...

## Kesimpulan
1. ... 2. ... 3. ... 4. ... 5. ...
```

---

## Checklist kelengkapan

- [ ] Judul + path notebook benar
- [ ] Dataset: sumber Kaggle, jumlah baris, distribusi kelas
- [ ] Rumus umum ditulis (LaTeX) sebelum dipakai
- [ ] Soal 1: langkah pengujian, akurasi, RMSE, baseline, confusion matrix
- [ ] Soal 2: data > 100 baris (ganti dataset via API)
- [ ] Soal 3: tabel data manual, hitung tiap rumus langkah demi langkah, akurasi/RMSE resubstitution
- [ ] Soal 4: bandingkan dengan metode lain pada data & fitur yang sama
- [ ] Soal 5: analisis metode mana lebih baik + alasan
- [ ] Semua angka hasil run kode asli (bukan karangan)
- [ ] Kesimpulan memuat semua poin soal
