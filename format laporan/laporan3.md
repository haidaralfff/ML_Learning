# Template Laporan Praktikum 3 — K-Means Clustering & K-Nearest Neighbors (K-NN)

Notebook: `notebook/{nama_notebook}.ipynb`

## Dataset

Dataset diambil dari Kaggle lewat API (`kagglehub`):

```python
path = kagglehub.dataset_download("{owner}/{dataset-slug}")
```

- Sumber: **{Nama Dataset}** ({asal/penyedia}), file `{nama_file}` ({ukuran}).
- Jumlah data: **{N} baris** (melebihi minimum {min}).
- Fitur numerik yang digunakan: `{kolom_fitur}`.
- Target / Label: `{kolom_target_atau_hasil_klaster}`.
- {Catatan pembersihan data: dropna, sampling baris, inisialisasi ID, dll.}

**Distribusi klaster / kelas:** {klaster_1} = {n_1} ({p_1}%), {klaster_2} = {n_2} ({p_2}%).

---

## Rumus umum perhitungan manual

Dipakai di seluruh laporan (soal 1, 3, 4, maupun soal 5).

**1. Jarak Euclidean (Euclidean Distance)**

$$ d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{m} (a_i - b_i)^2} $$

*Keterangan:* $\mathbf{a}$ dan $\mathbf{b}$ adalah vektor fitur berdimensi $m$.

**2. Pembaruan Titik Pusat Klaster (Centroid Update - K-Means)**

$$ \mathbf{c}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x} $$

*Keterangan:* $S_k$ adalah himpunan titik data yang menjadi anggota klaster $k$, dan $|S_k|$ adalah jumlah anggotanya.

**3. Klasifikasi K-Nearest Neighbors (K-NN Majority Voting)**

$$ \hat{y} = \arg\max_{c \in C} \sum_{i \in N_K(\mathbf{x})} I(y_i = c) $$

*Keterangan:* $N_K(\mathbf{x})$ adalah himpunan $K$ tetangga terdekat dari $\mathbf{x}$, dan $I(\cdot)$ adalah fungsi indikator bernilai 1 jika kondisi benar dan 0 jika salah.

**4. Klasifikasi Jarak ke Centroid (Nearest Centroid)**

$$ \hat{y} = \arg\min_{k} d(\mathbf{x}, \mathbf{c}_k) $$

*Keterangan:* Titik data $\mathbf{x}$ dimasukkan ke klaster yang memiliki jarak centroid $\mathbf{c}_k$ terkecil.

**5. Akurasi**

$$ \text{Accuracy} = \frac{\text{jumlah prediksi benar}}{n} \times 100\% $$

**6. RMSE (Root Mean Squared Error)**

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

---

## 1. Langkah pengujian — akurasi dan RMSE

### Langkah pengujian

1. {Melakukan klasterisasi data menggunakan K-Means untuk menghasilkan label klaster awal.}
2. {Acak urutan data menggunakan seed tertentu (`random.seed(42)`).}
3. {Split dataset menjadi **{persen_latih}% data latih ({n_latih})** dan **{persen_uji}% data uji ({n_uji})**.}
4. {Latih algoritma K-NN menggunakan data latih dan label hasil klaster K-Means.}
5. {Lakukan prediksi label klaster pada seluruh data uji.}
6. {Hitung nilai evaluasi Akurasi dan RMSE antara label acuan dan label hasil prediksi.}

### Hasil Pengujian

| Metrik | Nilai |
|---|---|
| Data latih | {n_latih} |
| Data uji | {n_uji} |
| Akurasi data uji | **{akurasi} ({akurasi_persen}%)** |
| RMSE data uji | **{rmse}** |

Konsistensi RMSE: error rate = $1 - {akurasi} = {error}$, maka $\text{RMSE} = \sqrt{{error}} \approx {rmse}$ ✓

**Confusion matrix pada data uji:**

| | Prediksi {Kelas_1} | Prediksi {Kelas_2} |
|---|---|---|
| **Sebenarnya {Kelas_1}** | {tp} (benar) | {fn} (salah) |
| **Sebenarnya {Kelas_2}** | {fp} (salah) | {tn} (benar) |

### Analisis Evaluasi: Akurasi, Overfitting, dan Karakteristik Data

1. **Penyebab Nilai Akurasi Tinggi / Sempurna:**
   - {Penjelasan kesamaan metrik jarak Euclidean antara K-Means dan K-NN (model surrogacy / distilasi).}
2. **Kajian Overfitting:**
   - {Perbedaan antara evaluasi resubstitution pada 20 data awal (menghafal) vs evaluasi train/test split pada data uji (generalisasi batas keputusan).}
3. **Pengaruh Karakteristik Data:**
   - {Pengaruh disparitas skala fitur yang belum dinormalisasi, separabilitas antar-klaster, dan kepadatan data latih.}
4. **Peran Konvergensi Iterasi:**
   - {Penjelasan bahwa hasil evaluasi stabil karena konvergensi K-Means dan K-NN sebagai lazy learner tanpa proses iterasi.}

---

## 2. Penggunaan data lain dengan jumlah lebih banyak (min. 100)

- Tugas praktikum sebelumnya menggunakan data awal sebanyak **{n_awal} baris**.
- Dataset diperbarui menggunakan dataset **{nama_dataset_baru}** via KaggleHub API sebanyak **{n_baru} baris** (melebihi batas minimum 100 data).
- {Alasan pemilihan dataset dan fitur yang digunakan.}

---

## 3. Perhitungan manual menggunakan 20 data pertama

Soal 3 dikerjakan **secara manual di laporan ini** menggunakan 20 baris data pertama (`x[:20]`) dari dataset yang sama. Seluruh hasil diverifikasi kesesuaiannya dengan program.

### 3.1 Tabel 20 Data Awal

| ID | {Fitur 1} | {Fitur 2} | {Fitur 3} | {Fitur 4} |
|---|---|---|---|---|
| P01 | {v} | {v} | {v} | {v} |
| ... | ... | ... | ... | ... |
| P20 | {v} | {v} | {v} | {v} |

Inisialisasi centroid awal:
- $\mathbf{c}_1 = [{c1_values}]$
- $\mathbf{c}_2 = [{c2_values}]$

### 3.2 Perhitungan Jarak dan Pengelompokan Iterasi 1

| ID | Data $\mathbf{x}$ | $d(\mathbf{x}, \mathbf{c}_1)$ | $d(\mathbf{x}, \mathbf{c}_2)$ | Klaster |
|---|---|---|---|---|
| P01 | [{v}] | {d1} | {d2} | {C} |
| ... | ... | ... | ... | ... |
| P20 | [{v}] | {d1} | {d2} | {C} |

### 3.3 Pembaruan Centroid (Iterasi 1)

**Centroid baru $\mathbf{c}_1$ ($n = {n_c1}$):**
$$ \mathbf{c}_1 = \left[ \frac{\sum f_1}{{n_c1}}, \frac{\sum f_2}{{n_c1}}, \frac{\sum f_3}{{n_c1}}, \frac{\sum f_4}{{n_c1}} \right] = [{c1_new}] $$

**Centroid baru $\mathbf{c}_2$ ($n = {n_c2}$):**
$$ \mathbf{c}_2 = \left[ \frac{\sum f_1}{{n_c2}}, \frac{\sum f_2}{{n_c2}}, \frac{\sum f_3}{{n_c2}}, \frac{\sum f_4}{{n_c2}} \right] = [{c2_new}] $$

### 3.4 Iterasi Lanjutan & Konvergensi

- Perhitungan jarak ulang ke centroid baru.
- Perpindahan anggota antar klaster.
- Centroid konvergen pada iterasi ke-{iterasi_konvergen}.

### 3.5 Analisis Hasil Program vs Perhitungan Manual

1. **Kesesuaian hasil:** {Penjelasan verifikasi kecocokan numerik program dengan hitungan manual.}
2. **Karakteristik klaster:** {Penjelasan profil segmen C1 dan C2.}

---

## 4. Program K-NN tanpa library ML menggunakan data hasil clustering

### Penjelasan Implementasi

1. **Representasi Data:** Data hasil clustering K-Means digunakan sebagai pasangan fitur $(\mathbf{x})$ dan label target $(y \in \{0, 1\})$.
2. **Perhitungan Jarak:** Menggunakan fungsi `euclidean(a, b)` murni dengan `math.sqrt`.
3. **Penentuan Tetangga Terdekat:** Mengurutkan seluruh jarak dari titik uji ke data latih dan memilih $K$ nilai terkecil (`k_neighbors = {k}`).
4. **Majority Voting:** Menggunakan `Counter.most_common(1)` untuk menentukan label mayoritas.

---

## 5. Perbandingan: Jarak ke Centroid vs Metode K-NN

### 5.1 Perbandingan Hasil Pengujian

| Parameter Evaluasi | Metode K-NN ($K={k}$) | Metode Jarak Centroid |
|---|---|---|
| Akurasi | **{acc_knn}%** | **{acc_cen}%** |
| RMSE | **{rmse_knn}** | **{rmse_cen}** |
| Waktu Eksekusi | {waktu_knn} detik | {waktu_cen} detik |
| Kompleksitas Waktu | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(K \cdot d)$ |

### 5.2 Analisis: Manakah yang Lebih Baik dan Mengapa?

1. **Dari Sudut Pandang Akurasi dan Bentuk Batas Keputusan (*Decision Boundary*):**
   - {Penjelasan mengapa K-NN lebih fleksibel menangkap pola lokal / non-linear dibandingkan batas linear/Voronoi centroid.}
2. **Dari Sudut Pandang Kompleksitas Komputasi dan Skalabilitas:**
   - {Penjelasan perbandingan efisiensi waktu inference antara membandingkan ke $N$ titik latih vs ke $K$ titik centroid.}
3. **Kesimpulan Pemilihan Metode:**
   - {Rekomendasi metode terbaik berdasarkan kondisi data dan kebutuhan aplikasi.}

---

## Kesimpulan

1. {Poin 1: Hasil pengujian akurasi dan RMSE model.}
2. {Poin 2: Penggunaan dataset >100 baris dari Kaggle.}
3. {Poin 3: Hasil perhitungan manual 20 data dan verifikasi program.}
4. {Poin 4: Keberhasilan implementasi K-NN scratch dari label K-Means.}
5. {Poin 5: Kesimpulan perbandingan K-NN vs Jarak Centroid.}
