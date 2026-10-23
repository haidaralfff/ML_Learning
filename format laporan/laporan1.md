# Laporan Praktikum 1 — Menggunakan Dataset eCommerce Customer Service Satisfaction 

Notebook: `notebook/{nama_notebook}.ipynb`

## Dataset

Dataset diambil dari Kaggle lewat API (`kagglehub`):

```python
path = kagglehub.dataset_download("{owner}/{dataset-slug}")
```

- Sumber: **{Nama Dataset}** ({asal/penyedia}), file `{nama_file}` ({ukuran}).
- Jumlah data: **{N} baris** (melebihi minimum {min}).
- Kolom yang dipakai: `{kolom_fitur}` (fitur), `{kolom_target}` (target).
- {Catatan pembersihan: handling kolom sisa, encoding file, dropna, dll.}

**Distribusi kelas/target:** {kelas_1} = {n_1} ({p_1}%), {kelas_2} = {n_2} ({p_2}%).

### Tabel encoding / mapping (jika fitur kategorikal)

| Fitur | Mapping |
|---|---|
| {fitur_1} | {nilai_a}={angka}, {nilai_b}={angka}, ... |
| {fitur_2} | {nilai_a}={angka}, {nilai_b}={angka}, ... |

---

## Rumus umum perhitungan manual

Dipakai di seluruh laporan (soal 1 maupun soal 3).

**1. {Rumus 1 — mis. Entropy}**

$$ {ekspresi\_rumus} $$

{penjelasan singkat simbol: p_i = ..., n = ..., dst.}

**2. {Rumus 2 — mis. Information Gain}**

$$ {ekspresi\_rumus} $$

**3. {Rumus 3 — mis. Prediksi / telusur pohon}**

$$ {ekspresi\_rumus} $$

**4. Akurasi**

$$\text{Accuracy} = \frac{\text{jumlah prediksi benar}}{n} \times 100\%$$

**5. RMSE**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

---

## 1. Langkah pengujian — akurasi dan RMSE

### Langkah pengujian

1. {Encode label jadi angka (jika perlu RMSE).}
2. Acak urutan data dengan seed tetap (`random.seed(42)`).
3. Split **{persen}% data latih ({n_latih})** dan **{persen}% data uji ({n_uji})**.
4. Bangun model dari **data latih saja** (`{fungsi_buat_model}`).
5. Prediksi tiap baris data uji (`{fungsi_prediksi}`).
6. Hitung akurasi dan RMSE pada data uji.

### Hasil

| Metrik | Nilai |
|---|---|
| Data latih | {n_latih} |
| Data uji | {n_uji} |
| Akurasi data uji | **{akurasi} ({akurasi_persen}%)** |
| RMSE data uji | **{rmse}** |

Konsistensi RMSE: error rate = $1 - {akurasi} = {error}$, maka $\text{RMSE} = \sqrt{{error}} \approx {rmse}$ ✓

**Baseline** "tebak {kelas_mayoritas} semua" = {baseline_persen}%. Model ({akurasi_persen}%) {melampaui/setara} baseline.

**Confusion matrix pada data uji:**

| | Prediksi {positif} | Prediksi {negatif} |
|---|---|---|
| **Sebenarnya {positif}** | {tp} (benar) | {fn} (salah) |
| **Sebenarnya {negatif}** | {fp} (salah) | {tn} (benar) |

{Analisis singkat confusion: FN vs FP, kaitan dengan prior/ketidakseimbangan kelas.}

---

## 2. Penggunaan data lain dengan jumlah lebih banyak (min. {min})

- Tugas sebelumnya memakai dataset {dataset_lama} dengan **{n_lama} baris**.
- Data diganti ke **{nama_dataset_baru}** dengan **{n_baru} baris** (melebihi minimum {min}).
- {Cara baca data: nrows sampling, encoding, dll.}

---

## 3. Perhitungan manual dengan {n} data

Soal 3 dikerjakan **secara manual di laporan ini**, memakai {n} baris pertama dari data yang sama. Semua hasil sudah diverifikasi dengan program.

### 3.1 Data yang dipakai ({n} baris pertama)

| # | {fitur_1} | {fitur_2} | ... | {target} |
|---|---|---|---|---|
| 1 | {v} | {v} | ... | {label} |
| ... | ... | ... | ... | ... |
| {n} | {v} | {v} | ... | {label} |

Jumlah: **{kelas_1} = {n_1}, {kelas_2} = {n_2}, $N = {n}$**.

### 3.2 {Rumus 1} (manual)

$$ {perhitungan\_dengan\_angka} = {hasil} $$

### 3.3 {Rumus 2} (manual, per fitur per kelas)

**Kelas {kelas_1} ($n = {n_1}$):**

| Fitur | Nilai | Hitung | {rumus} |
|---|---|---|---|
| {fitur} | {nilai} | {a}/{n_1} | {hasil} |

**Kelas {kelas_2} ($n = {n_2}$):**

| Fitur | Nilai | Hitung | {rumus} |
|---|---|---|---|
| {fitur} | {nilai} | {a}/{n_2} | {hasil} |

### 3.4 Perhitungan manual (contoh beberapa data / struktur hasil)

**Data #{i}** — $\mathbf{x} = [...]$, asli {label}:

$$ {perhitungan} = {hasil} $$
$$ \Rightarrow {kesimpulan\_prediksi} \Rightarrow \textbf{prediksi (benar/salah)} $$

{Ulangi untuk beberapa data representatif, atau tampilkan struktur hasil (mis. pohon) + penelusuran 2-3 data.}

### 3.5 Pengujian ulang {n} data (resubstitution)

Salah prediksi pada **data #{a} dan #{b}** ({k} dari {n}):

$$\text{Akurasi} = \frac{{n}-{k}}{{n}} = {akurasi}\;({persen}\%), \qquad \text{RMSE} = \sqrt{\frac{{k}}{{n}}} = {rmse}$$

### 3.6 Analisis hasil ({n} data vs data besar)

1. **{Overfitting / underfitting / kelas tidak seimbang}.** {Penjelasan.}
2. **{Perbandingan akurasi resubstitution vs data uji}.** {Penjelasan.}
3. **Kesimpulan perbandingan:** {apa yang terlihat dari perbedaan data kecil vs besar.}

---

## Kesimpulan

1. {Poin 1 — hasil pengujian utama + perbandingan baseline.}
2. {Poin 2 — penggantian data & jumlah.}
3. {Poin 3 — hasil perhitungan manual + analisis.}
