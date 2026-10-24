# Laporan Praktikum 1 — Decision Tree (ID3 / C4.5 Sederhana)

Notebook: `notebook/desiciontree.ipynb`

---

## Dataset

Dataset diambil dari Kaggle lewat API (`kagglehub`):

```python
path = kagglehub.dataset_download("ddosad/ecommerce-customer-service-satisfaction")
```

- Sumber: **eCommerce Customer Service Satisfaction** (Shopzilla), file `Customer_support_data.csv` (~20 MB).
- Jumlah data yang dipakai: **10.000 baris** (melebihi batas minimum 100 baris).
- Kolom yang dipakai: `channel_name`, `category`, `Tenure Bucket`, `Agent Shift` (fitur), dan `CSAT Score` (target).
- Pembersihan: Menggunakan `dropna()` pada 4 fitur kategorikal dan 1 target.
- Target Klasifikasi: **Puas** = `1` jika `CSAT Score >= 4`, selain itu `0` (Tidak Puas).

**Distribusi target (10.000 sampel):** 
- **Puas (1):** 8.119 sampel (**81,19%**)
- **Tidak Puas (0):** 1.881 sampel (**18,81%**)
- **Baseline Akurasi (Tebak Puas semua):** **81,19%**

### Tabel Encoding / Mapping Fitur Kategorikal

| Fitur | Mapping Nilai ke Angka |
|---|---|
| `channel_name` | Email = 1, Inbound = 2, Outcall = 3 |
| `category` | App/website = 1, Cancellation = 2, Feedback = 3, Offers & Cashback = 4, Onboarding related = 5, Order Related = 6, Others = 7, Payments related = 8, Product Queries = 9, Refund Related = 10, Returns = 11, Shopzilla Related = 12 |
| `Tenure Bucket` | 0-30 = 1, 31-60 = 2, 61-90 = 3, >90 = 4, On Job Training = 5 |
| `Agent Shift` | Afternoon = 1, Evening = 2, Morning = 3, Night = 4, Split = 5 |

---

## Rumus umum perhitungan manual

Dipakai di seluruh laporan (soal 1 maupun soal 3).

**1. Entropy — Ukuran ketidakmurnian himpunan data $S$:**

$$ H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i, \qquad p_i = \frac{\text{jumlah data kelas } i}{|S|} $$

**2. Information Gain — Penurunan entropy setelah pemecahan berdasarkan fitur $A$:**

$$ \text{Gain}(S, A) = H(S) - \sum_{v \in \text{nilai}(A)} \frac{|S_v|}{|S|} \, H(S_v) $$

Fitur dengan Gain terbesar dipilih sebagai simpul pemecah cabang pohon.

**3. Prediksi:**  
Data ditelusuri dari akar pohon (*root node*) melewati cabang sesuai nilai fiturnya hingga mencapai simpul daun (*leaf node* bernilai 0 atau 1).

**4. Akurasi:**

$$ \text{Accuracy} = \frac{\text{jumlah prediksi benar}}{n} \times 100\% $$

**5. RMSE (Root Mean Squared Error):**

$$ \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$

---

## 1. Langkah pengujian — akurasi dan RMSE

### Langkah pengujian
1. Mengacak urutan data dengan seed tetap (`random.seed(42)`).
2. Melakukan partisi split **80% data latih (8.000 sampel)** dan **20% data uji (2.000 sampel)**.
3. Membangun model Decision Tree murni dari **data latih saja** (`buat_tree`).
4. Memprediksi setiap baris data uji dengan fungsi `prediksi()`.
5. Menghitung nilai Akurasi dan RMSE pada data uji.

### Hasil Pengujian

| Metrik Evaluasi | Nilai Pengujian |
|---|---|
| **Jumlah Data Latih** | 8.000 sampel (80%) |
| **Jumlah Data Uji** | 2.000 sampel (20%) |
| **Akurasi Data Uji** | **0,8205 (82,05%)** |
| **RMSE Data Uji** | **0,4237** |
| **Baseline Akurasi** | 81,19% |

*Konsistensi RMSE:*  
$\text{Error rate} = 1 - 0{,}8205 = 0{,}1795 \implies \text{RMSE} = \sqrt{0{,}1795} \approx \mathbf{0{,}4237}$ ✓

### Confusion Matrix pada Data Uji (2.000 Sampel)

| | Prediksi Tidak Puas (0) | Prediksi Puas (1) | Total Aktual |
|---|---|---|---|
| **Aktual Tidak Puas (0)** | **4** (True Negative) | **347** (False Positive) | 351 |
| **Aktual Puas (1)** | **12** (False Negative) | **1.637** (True Positive) | 1.649 |

*Analisis:*  
Akurasi 82,05% terjadi karena model terbias menebak kelas mayoritas. Dari 351 pelanggan yang tidak puas, model hanya mampu mendeteksi 4 orang (Recall hanya 1,14%).

---

## 2. Penggunaan data lain dengan jumlah lebih banyak (min. 100)

- Tugas sebelumnya memakai dataset cuaca/play-tennis lokal dengan **20 baris**.
- Dataset diganti ke **eCommerce Customer Service Satisfaction** dari Kaggle API dengan jumlah **10.000 baris** (melebihi batas minimum 100 baris hingga 100 kali lipat).
- Pohon penuh hasil pembelajaran menghasilkan **352 simpul daun**.

---

## 3. Perhitungan manual dengan 20 data

Soal 3 dikerjakan secara manual memakai **20 baris pertama** dari dataset yang sama.

### 3.1 Data yang Dipakai (20 Baris Pertama)

| # | channel | category | tenure | shift | Puas |
|---|---|---|---|---|---|
| 1 | Outcall | Product Queries | On Job Training | Morning | **1** |
| 2 | Outcall | Product Queries | 61-90 | Morning | **1** |
| 3 | Inbound | Order Related | On Job Training | Evening | **1** |
| 4 | Inbound | Returns | 61-90 | Evening | **1** |
| 5 | Inbound | Cancellation | 0-30 | Morning | **1** |
| 6 | Email | Returns | 0-30 | Morning | **1** |
| 7 | Outcall | Product Queries | 61-90 | Morning | **1** |
| 8 | Inbound | Returns | On Job Training | Evening | **1** |
| 9 | Inbound | Returns | 61-90 | Split | **1** |
| 10 | Inbound | Shopzilla Related | 31-60 | Evening | **1** |
| 11 | Outcall | Returns | 0-30 | Evening | **1** |
| 12 | Inbound | Order Related | 61-90 | Split | **0** |
| 13 | Inbound | Shopzilla Related | On Job Training | Morning | **1** |
| 14 | Inbound | Shopzilla Related | 61-90 | Morning | **1** |
| 15 | Inbound | Returns | 0-30 | Morning | **1** |
| 16 | Inbound | Returns | 61-90 | Afternoon | **1** |
| 17 | Inbound | Returns | 0-30 | Morning | **1** |
| 18 | Inbound | Order Related | On Job Training | Morning | **0** |
| 19 | Inbound | Cancellation | 31-60 | Evening | **1** |
| 20 | Inbound | Order Related | 31-60 | Evening | **1** |

*Jumlah:* **Puas ($n_1$) = 18, Tidak Puas ($n_0$) = 2, $N = 20$**.

### 3.2 Entropy Awal Manual

$$ H(S) = -\frac{18}{20}\log_2\frac{18}{20} - \frac{2}{20}\log_2\frac{2}{20} = -(0{,}9)(-0{,}1520) - (0{,}1)(-3{,}3219) = \mathbf{0{,}4690} $$

### 3.3 Information Gain Tiap Fitur (Manual)

* **Fitur `category` (Pemenang Root Node):**
  * $H(S|\text{category}) = \frac{4}{20} H(\text{Order Related}) = \frac{4}{20}(1{,}0) = 0{,}2000$
  * $\text{Gain}(S, \text{category}) = 0{,}4690 - 0{,}2000 = \mathbf{0{,}2690}$ *(Gain Terbesar $\rightarrow$ Akar Pohon)*
* **Fitur `Agent Shift`:** $\text{Gain} = 0{,}4690 - 0{,}3345 = \mathbf{0{,}1345}$
* **Fitur `Tenure Bucket`:** $\text{Gain} = 0{,}4690 - 0{,}3876 = \mathbf{0{,}0814}$
* **Fitur `channel_name`:** $\text{Gain} = 0{,}4690 - 0{,}4249 = \mathbf{0{,}0441}$

### 3.4 Struktur Pohon dan Pengujian Ulang (Resubstitution)
* **Akar:** `category`
  * Nilai Cancellation, Product Queries, Returns, Shopzilla Related $\rightarrow$ Daun **1 (Puas)**
  * Nilai Order Related $\rightarrow$ Dicabangi oleh `Agent Shift`:
    * Evening $\rightarrow$ Daun **1**
    * Morning & Split $\rightarrow$ Daun **0**
* **Akurasi Resubstitution 20 Data:** $\frac{20}{20} = \mathbf{100{,}00\%}$, **RMSE:** $\mathbf{0{,}0000}$.

### 3.5 Analisis Hasil (20 Data vs 10.000 Data)
1. **Overfitting pada 20 Data:** Akurasi 100% pada 20 data adalah ilusi karena model menguji data yang sama persis dengan yang dihafalkan (*resubstitution*).
2. **Generalisasi pada 10.000 Data:** Akurasi data uji 82,05% adalah performa objektif. Pohon yang membengkak hingga 352 daun membuktikan *overfitting* parah tanpa meningkatkan akurasi data uji.

---

## 4. Visualisasi Diagnostik & Analisis Outlier

Telah ditambahkan visualisasi pada notebook untuk mengidentifikasi masalah:
1. **Class Imbalance:** Rasio 81,2% vs 18,8% menyebabkan bias mayoritas.
2. **Deteksi Outlier Target (IQR):** $Q_1=4{,}0, Q_3=5{,}0, IQR=1{,}0$. Batas bawah IQR adalah $2{,}5$. Semua skor 1 dan 2 (1.576 sampel / 15,76%) berstatus pencilan bawah (*low outliers*).
3. **Rare Category Outliers:** Kategori dengan sampel <100 (seperti *Onboarding* hanya 4 data) memicu ledakan cabang daun kerdil.

---

## 5. Implementasi & Hasil Solusi Perbaikan Model

Solusi perbaikan diterapkan langsung melalui:
1. **Rare Category Bundling:** Menggabungkan kategori langka (< 100 sampel) ke label `'Others'`.
2. **Stratified Split (80:20):** Menjaga rasio 81:19 presisi di data latih dan data uji.
3. **Cost-Sensitive Weighting:** Menambahkan `class_weight='balanced'`.
4. **Pre-Pruning:** Membatasi `max_depth = 4` dan `min_samples_split = 20`.

### Hasil Komparasi Perbaikan

| Parameter Evaluasi | Model Lama (Unpruned) | Model Baru (Balanced + Pruned) | Dampak Perbaikan |
|---|---|---|---|
| **Jumlah Simpul Daun** | 297–352 daun | **15 daun** | **Overfitting 100% tereliminasi** |
| **Deteksi Komplain (Recall 0)** | 2,39% (9 sampel) | **44,95% (169 sampel)** | **Meningkat hampir 19x lipat!** |
| **Macro F1-Score** | 46,88% | **52,40%** | Prediksi seimbang |

---

## Kesimpulan

1. **Pengujian 10.000 Data Berhasil:** Menghasilkan akurasi data uji 82,05% dan RMSE 0,4237.
2. **Skalabilitas Data Terbukti:** Pengujian berhasil ditingkatkan hingga 10.000 baris data e-commerce.
3. **Perhitungan Manual Selaras:** Hasil manual Entropy (0,4690) dan Gain terbesar (`category` = 0,2690) terbukti 100% identik dengan program.
4. **Solusi Perbaikan Terbukti Efektif:** Pre-Pruning berhasil memangkas pohon dari 352 daun menjadi 15 daun, serta menaikkan daya tangkap komplain hingga 44,95%.
