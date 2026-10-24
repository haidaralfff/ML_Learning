# Laporan 1 — Decision Tree (ID3 / C4.5 sederhana)

Notebook: `notebook/desiciontree.ipynb`

---

## Dataset

Dataset diambil dari Kaggle lewat API (`kagglehub`):

```python
path = kagglehub.dataset_download("ddosad/ecommerce-customer-service-satisfaction")
```

- Sumber: **eCommerce Customer Service Satisfaction** (Shopzilla), file `Customer_support_data.csv` (~20 MB).
- Jumlah data yang dipakai: **10.000 baris** (sampling dari 85.907 baris).
- Fitur kategorikal (4): `channel_name`, `category`, `Tenure Bucket`, `Agent Shift`.
- Target: **Puas** = `1` jika `CSAT Score >= 4`, selain itu `0`.

Encoding kategorikal → angka (kamus dibangun dari nilai unik tiap kolom, urut):

| Fitur | Mapping |
|---|---|
| channel_name | Email=1, Inbound=2, Outcall=3 |
| category | App/website=1, Cancellation=2, Feedback=3, Offers & Cashback=4, Onboarding related=5, Order Related=6, Others=7, Payments related=8, Product Queries=9, Refund Related=10, Returns=11, Shopzilla Related=12 |
| Tenure Bucket | 0-30=1, 31-60=2, 61-90=3, >90=4, On Job Training=5 |
| Agent Shift | Afternoon=1, Evening=2, Morning=3, Night=4, Split=5 |

Distribusi kelas pada 10.000 data: **Puas = 8.119 (81,19%)**, Tidak Puas = 1.881 (18,81%).

---

## Rumus umum perhitungan manual

Dipakai di seluruh laporan (soal 1 maupun soal 3).

**1. Entropy** — ukuran ketidakmurnian sebuah himpunan data $S$ dengan $c$ kelas:

$$H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i, \qquad p_i = \frac{\text{jumlah data kelas } i}{|S|}$$

Contoh dua kelas: $H(S) = -\frac{n_1}{n}\log_2\frac{n_1}{n} - \frac{n_0}{n}\log_2\frac{n_0}{n}$.
Nilai $0 \le H(S) \le \log_2 c$; murni (satu kelas saja) → $H = 0$.

**2. Information Gain** — penurunan entropy setelah data dibagi per nilai fitur $A$:

$$\text{Gain}(S, A) = H(S) - H(S|A) = H(S) - \sum_{v \in \text{nilai}(A)} \frac{|S_v|}{|S|} \, H(S_v)$$

dengan $S_v$ = himpunan data yang memiliki $A = v$.
Fitur dengan **Gain terbesar** dipilih menjadi simpul pembagian (akar/cabang), lalu proses diulang rekursif pada tiap subset sampai subset murni atau fitur habis.

**3. Prediksi** — satu data dijalankan dari akar pohon: pada tiap simpul, cocokkan nilai fitur dengan cabangnya, lanjut sampai mencapai **leaf** (nilai 0 atau 1).

**4. Akurasi**:

$$\text{Accuracy} = \frac{\text{jumlah prediksi benar}}{n} \times 100\%$$

**5. RMSE** (label dan prediksi sama-sama 0/1):

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Untuk klasifikasi biner 0/1, RMSE = $\sqrt{\text{error rate}}$; RMSE = 0 berarti semua prediksi benar.

---

## 1. Langkah pengujian — akurasi dan RMSE

### Langkah pengujian

1. Acak urutan data dengan seed tetap (`random.seed(42)`).
2. Split **80% data latih (8.000)** dan **20% data uji (2.000)**.
3. Bangun pohon keputusan dari **data latih saja** (`buat_tree`).
4. Prediksi setiap baris data uji dengan fungsi `prediksi()` (telusuri pohon dari akar).
5. Hitung akurasi dan RMSE pada data uji dengan rumus no. 4 dan 5 di atas.

### Hasil

| Metrik | Nilai |
|---|---|
| Data latih | 8.000 |
| Data uji | 2.000 |
| Akurasi data uji | **0,8205 (82,05%)** |
| RMSE data uji | **0,4237** |

Perhitungan RMSE dari akurasi (konsistensi): error rate = $1 - 0{,}8205 = 0{,}1795$, maka $\text{RMSE} = \sqrt{0{,}1795} \approx 0{,}4237$ ✓

Catatan: karena kelas tidak seimbang (81% puas), **baseline** "tebak semua puas" hanya memberi akurasi 81,19% dengan RMSE $\sqrt{0{,}1881} \approx 0{,}4337$. Model decision tree (82,05% / 0,4237) sedikit lebih baik dari baseline — fitur `category` dan `Agent Shift` hanya memberi tambahan informasi terbatas untuk dataset ini.

---

## 2. Visualisasi Diagnostik Masalah Model & Indikasi Overfitting

Untuk membuktikan secara visual akar permasalahan dari model Decision Tree pada dataset ini, telah ditambahkan **Plot Diagnostik 4 Panel** pada notebook:

1. **Panel 1: Ketimpangan Kelas (*Class Imbalance*)**
   - Menampilkan perbandingan ekstrem antara pelanggan **Puas (81,2%)** vs **Tidak Puas (18,8%)**.
   - Garis horizontal putus-putus memperlihatkan batas **Baseline Akurasi (81,2%)**. Ini membuktikan secara visual bahwa akurasi 82% yang dicapai model bukanlah prestasi spektakuler, melainkan akibat dari model yang terbias menebak kelas mayoritas.

2. **Panel 2: Peringkat Information Gain Fitur**
   - Menggambarkan seberapa besar informasi yang disumbangkan oleh masing-masing fitur:
     - `category` memiliki Information Gain tertinggi (~0,027 bit) sehingga konsisten terpilih sebagai simpul akar (*root node*).
     - `channel_name`, `Tenure Bucket`, dan `Agent Shift` memiliki gain sangat rendah (< 0,005 bit). Ini menjelaskan mengapa pohon sulit memecah kelas secara murni tanpa menghafal kombinasi langka.

3. **Panel 3: Heatmap Confusion Matrix pada Data Uji**
   - Memperlihatkan bahwa dari 2.000 data uji:
     - Kelas Puas (1) mayoritas tertebak dengan benar (>1.600 sampel).
     - Kelas Tidak Puas (0) sebagian besar gagal dideteksi dan salah ditebak sebagai Puas karena pohon terlalu terdominasi oleh kelas mayoritas.

4. **Panel 4: Kurva Kompleksitas Daun vs Akurasi Uji (*Bukti Kuat Overfitting*)**
   - Menguji pertumbuhan pohon dari `max_depth = 1` (3 daun), `depth = 2` (11 daun), `depth = 3` (33 daun), `depth = 4` (88 daun) hingga **Pohon Penuh / Unpruned (330–352 daun)**.
   - **Temuan Kunci:** Akurasi data uji tetap mendatar pada kisaran **81,5% – 82,0%**.
   - **Kesimpulan Overfitting:** Menumbuhkan pohon hingga 352 daun sama sekali tidak meningkatkan performa generalisasi data uji. Ratusan daun tersebut murni menghafal data pencilan (*noise*). Oleh karena itu, penerapan **Pre-Pruning (`max_depth = 3` atau `4`)** adalah solusi yang paling tepat untuk memangkas kompleksitas tanpa mengorbankan akurasi.

### 2.1 Deteksi & Analisis Outlier Data

Pemeriksaan data pencilan (*outlier*) dilakukan menggunakan metode statistik **Interquartile Range (IQR / Tukey's Rule)** dan analisis sebaran frekuensi:

1. **Outlier Target (CSAT Score) — 1.576 Sampel (15,76%):**
   - $Q_1 = 4,0$, $Q_3 = 5,0$, $IQR = 1,0$.
   - Batas Bawah IQR = $Q_1 - 1,5 \times IQR = 4,0 - 1,5 = \mathbf{2,5}$.
   - Seluruh pelanggan yang memberi **Skor 1 (1.433 sampel)** dan **Skor 2 (143 sampel)** secara matematis berstatus **pencilan bawah (*low outliers*)**. Hal ini menjelaskan mengapa pohon mengabaikan pelanggan komplain karena sebaran data terpusat di skor 4 dan 5 (81,2%).

2. **Outlier Frekuensi (*Rare Category Outliers* pada `category`):**
   - Ditemukan 4 kategori dengan sampel sangat langka (<100 sampel): *Offers & Cashback* (78), *App/website* (17), *Others* (10), dan *Onboarding related* (hanya 4 sampel).
   - Kategori-kategori ini menyebabkan algoritma ID3 membuat cabang-cabang kerdil yang memicu pembengkakan daun (*overfitting*).

3. **Outlier Fitur Numerik (`Item_price`):**
   - Median harga barang adalah Rp 1.016 dengan batas atas IQR Rp 11.103. Terdapat 96 sampel pencilan ekstrem dengan harga melonjak hingga Rp 108.490.

---

## 3. Penggunaan data lain dengan jumlah lebih banyak (min. 100)

- Tugas praktikum sebelumnya memakai dataset cuaca/play-tennis lokal dengan **20 baris**.
- Pada laporan ini data diganti ke dataset Kaggle lain (**eCommerce Customer Service Satisfaction**) dan jumlahnya dinaikkan menjadi **10.000 baris** (jauh melebihi minimum 100).
- Pembacaan sampling: `pd.read_csv(data, nrows=10000)`.
- Dengan data sebesar ini, pohon hasil `buat_tree` memiliki **352 simpul daun** — jauh lebih kompleks dibanding pohon 20 data (5 daun), dan waktu training tetap ~0,1 detik.

---

## 4. Perhitungan manual dengan 20 data

Soal 3 dikerjakan **secara manual di laporan ini** (tanpa cell program terpisah), memakai 20 baris pertama dari data yang sama. Semua hasil di bawah sudah diverifikasi dengan perhitungan program `entropy()` dan `gain()`.

### 4.1 Data yang dipakai (20 baris pertama)

| # | channel | category | tenure | shift | Puas |
|---|---|---|---|---|---|
| 1 | Outcall | Product Queries | On Job Training | Morning | 1 |
| 2 | Outcall | Product Queries | 61-90 | Morning | 1 |
| 3 | Inbound | Order Related | On Job Training | Evening | 1 |
| 4 | Inbound | Returns | 61-90 | Evening | 1 |
| 5 | Inbound | Cancellation | 0-30 | Morning | 1 |
| 6 | Email | Returns | 0-30 | Morning | 1 |
| 7 | Outcall | Product Queries | 61-90 | Morning | 1 |
| 8 | Inbound | Returns | On Job Training | Evening | 1 |
| 9 | Inbound | Returns | 61-90 | Split | 1 |
| 10 | Inbound | Shopzilla Related | 31-60 | Evening | 1 |
| 11 | Outcall | Returns | 0-30 | Evening | 1 |
| 12 | Inbound | Order Related | 61-90 | Split | **0** |
| 13 | Inbound | Shopzilla Related | On Job Training | Morning | 1 |
| 14 | Inbound | Shopzilla Related | 61-90 | Morning | 1 |
| 15 | Inbound | Returns | 0-30 | Morning | 1 |
| 16 | Inbound | Returns | 61-90 | Afternoon | 1 |
| 17 | Inbound | Returns | 0-30 | Morning | 1 |
| 18 | Inbound | Order Related | On Job Training | Morning | **0** |
| 19 | Inbound | Cancellation | 31-60 | Evening | 1 |
| 20 | Inbound | Order Related | 31-60 | Evening | 1 |

Jumlah: **Puas ($n_1$) = 18, Tidak Puas ($n_0$) = 2, $n = 20$**.

### 4.2 Entropy awal (manual)

$$H(S) = -\frac{18}{20}\log_2\frac{18}{20} - \frac{2}{20}\log_2\frac{2}{20}$$
$$= -(0,9)\log_2(0,9) - (0,1)\log_2(0,1)$$
$$= -(0,9)(-0,1520) - (0,1)(-3,3219) = 0,1368 + 0,3322 = \mathbf{0,4690}$$

### 4.3 Information gain tiap fitur (manual)

Pakai rumus $\text{Gain}(S, A) = H(S) - \sum_v \frac{|S_v|}{20} H(S_v)$.

#### Fitur `category` (gain terbesar → akar pohon)

| Nilai category | $|S_v|$ | Label | $H(S_v)$ |
|---|---|---|---|
| Cancellation (2) | 2 | 1,1 | 0 |
| Order Related (6) | 4 | 1,0,0,1 | 1,0000 |
| Product Queries (9) | 3 | 1,1,1 | 0 |
| Returns (11) | 8 | 1×8 | 0 |
| Shopzilla Related (12) | 3 | 1,1,1 | 0 |

$H(S_6) = -\frac{2}{4}\log_2\frac{2}{4} - \frac{2}{4}\log_2\frac{2}{4} = 0,5 + 0,5 = 1{,}0000$

$$H(S|\text{category}) = \frac{4}{20}(1,0) = 0,2000$$
$$\text{gain} = 0,4690 - 0,2000 = \mathbf{0,2690}$$

#### Fitur `Agent Shift`

| Nilai shift | $|S_v|$ | Label | $H(S_v)$ |
|---|---|---|---|
| Afternoon (1) | 1 | 1 | 0 |
| Evening (2) | 7 | 1×7 | 0 |
| Morning (3) | 10 | 1×9, 0×1 | 0,4690 |
| Split (5) | 2 | 1,0 | 1,0000 |

$H(S_3) = -\frac{9}{10}\log_2\frac{9}{10} - \frac{1}{10}\log_2\frac{1}{10} = 0,9(0,1520) + 0,1(3,3219) = 0{,}4690$

$$H(S|\text{shift}) = \frac{10}{20}(0,4690) + \frac{2}{20}(1,0) = 0,2345 + 0,1000 = 0,3345$$
$$\text{gain} = 0,4690 - 0,3345 = \mathbf{0,1345}$$

#### Fitur `Tenure Bucket`

| Nilai tenure | $|S_v|$ | Label | $H(S_v)$ |
|---|---|---|---|
| 0-30 (1) | 5 | 1×5 | 0 |
| 31-60 (2) | 3 | 1×3 | 0 |
| 61-90 (4) | 7 | 1×6, 0×1 | 0,5917 |
| On Job Training (5) | 5 | 1×4, 0×1 | 0,7219 |

$H(S_4) = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = 0{,}5917$
$H(S_5) = -\frac{4}{5}\log_2\frac{4}{5} - \frac{1}{5}\log_2\frac{1}{5} = 0{,}7219$

$$H(S|\text{tenure}) = \frac{7}{20}(0,5917) + \frac{5}{20}(0,7219) = 0,2071 + 0,1805 = 0,3876$$
$$\text{gain} = 0,4690 - 0,3876 = \mathbf{0,0814}$$

#### Fitur `channel_name`

| Nilai channel | $|S_v|$ | Label | $H(S_v)$ |
|---|---|---|---|
| Email (1) | 1 | 1 | 0 |
| Inbound (2) | 15 | 1×13, 0×2 | 0,5665 |
| Outcall (3) | 4 | 1×4 | 0 |

$H(S_2) = -\frac{13}{15}\log_2\frac{13}{15} - \frac{2}{15}\log_2\frac{2}{15} = 0{,}5665$

$$H(S|\text{channel}) = \frac{15}{20}(0,5665) = 0,4249$$
$$\text{gain} = 0,4690 - 0,4249 = \mathbf{0,0441}$$

#### Peringkat gain

| Fitur | Gain |
|---|---|
| **category** | **0,2690** ← akar pohon |
| Agent Shift | 0,1345 |
| Tenure Bucket | 0,0814 |
| channel_name | 0,0441 |

### 4.4 Pembentukan Pohon 20 Data

Akar: **category**.
- Cancellation (2) → daun **1** (2 data murni)
- Product Queries (9) → daun **1** (3 data murni)
- Returns (11) → daun **1** (8 data murni)
- Shopzilla Related (12) → daun **1** (3 data murni)
- Order Related (6) → 4 data (2 puas, 2 tidak) → dicabangi fitur terbaik berikutnya: **Agent Shift**:
  - Evening (2) → daun **1** (P3, P20)
  - Morning (3) → daun **0** (P18)
  - Split (5) → daun **0** (P12)

Total: **5 cabang dari akar, menghasilkan 7 daun**.

### 4.5 Analisis Hasil (20 data vs 10.000 data)

| Parameter | 20 Data Pertama (Manual) | 10.000 Data (Data Uji 2.000) |
|---|---|---|
| **Akurasi** | **100,00%** (Resubstitution) | **82,05%** (Data Uji Terpisah) |
| **RMSE** | **0,0000** | **0,4237** |
| **Jumlah Daun** | 7 daun | 352 daun |
| **Karakter Evaluasi** | Menghafal data (*overfitting*) | Jujur dan menguji generalisasi |

---

## Kesimpulan

1. **Langkah Pengujian Berhasil:** Implementasi Decision Tree ID3 dari *scratch* pada 10.000 data menghasilkan akurasi **82,05%** dan RMSE **0,4237** pada data uji terpisah (80/20 split).
2. **Skalabilitas Data Terbukti:** Pengujian berhasil ditingkatkan dari 20 data cuaca menjadi **10.000 data riil** layanan pelanggan e-commerce via Kaggle API.
3. **Penyebab Overfitting Teridentifikasi:**
   - Pohon penuh (*unpruned*) menghasilkan hingga **352 simpul daun** untuk 4 fitur, namun akurasi uji tidak meningkat dibanding pohon sederhana.
   - Ketimpangan kelas (*class imbalance* 81% puas) membuat akurasi tampak tinggi karena mengikuti baseline tebakan mayoritas.
4. **Metode Penanganan yang Pas:**
   - Menerapkan **Pre-Pruning (`max_depth = 3` atau `4` dan `min_samples_split = 20`)** untuk menghentikan cabang liar.
   - Menerapkan **Stratified 80:20 Split** untuk menjaga proporsi kelas minoritas.
   - Menggunakan **Confusion Matrix** sebagai metrik penilai objektif.
5. **Perhitungan Manual 100% Selaras:** Perhitungan manual entropy (0,4690) dan gain terbesar pada fitur `category` (0,2690) terbukti identik dengan keluaran algoritma di notebook.

---

## 5. Implementasi & Hasil Solusi Perbaikan Model

Solusi perbaikan telah diterapkan langsung pada notebook melalui 4 tahapan:
1. **Rare Category Bundling:** Menggabungkan kategori langka (< 100 sampel) ke label `'Others'`.
2. **Stratified Split (80:20):** Memastikan rasio 81:19 konsisten di data latih dan data uji.
3. **Penyeimbangan Kelas (*Cost-Sensitive Weighting*):** Menambahkan `class_weight='balanced'`.
4. **Pre-Pruning:** Membatasi `max_depth = 4` dan `min_samples_split = 20`.

### Hasil Komparasi (Model Lama vs Model Baru)

| Parameter Evaluasi | Model Lama (Unpruned) | Model Baru (Balanced + Pruned) | Dampak Perbaikan |
|---|---|---|---|
| **Jumlah Simpul Daun** | 297–352 daun | **15 daun** | Mengeliminasi *overfitting* 100% |
| **Deteksi Komplain (Recall 0)** | 2,39% (9 dari 376 sampel) | **44,95% (169 dari 376 sampel)** | **Meningkat hampir 19x lipat!** |
| **Macro F1-Score** | 46,88% | **52,40%** | Prediksi jauh lebih seimbang |
| **Karakter Model** | Menghafal & bias kelas 1 | **Objektif mendeteksi tiket komplain** | Relevan untuk kebutuhan bisnis CS |

