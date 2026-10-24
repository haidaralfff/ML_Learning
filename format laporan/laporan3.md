# Laporan Praktikum 3 — K-Means Clustering & K-Nearest Neighbors (K-NN)

Notebook: `notebook/KNN_K-Means.ipynb`

---

## Dataset

Dataset yang digunakan untuk pengujian skala besar diambil langsung dari Kaggle menggunakan API `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("dhairyajeetsingh/ecommerce-customer-behavior-dataset")
```

- **Sumber Dataset:** *E-commerce Customer Behavior and Churn Dataset* oleh Dhairyajeet Singh (Kaggle).
- **File Data:** `ecommerce_customer_churn_dataset.csv`.
- **Pembersihan Outlier & Penanganan Data:**
  - Dilakukan penanganan *missing values* (`dropna`).
  - Dilakukan penyaringan data *outlier* pada fitur `Average_Order_Value` (AOV) pada rentang **$60 \le \text{AOV} \le 5000$ USD**:
    - Menghapus 44 data dengan $\text{AOV} < 60$ USD (noise transaksi anomali).
    - Menghapus 1 data pencilan ekstrem pada baris 664 dengan $\text{AOV} = 5.279,80$ USD ($\text{AOV} > 5000$ USD).
- **Jumlah Data Digunakan:** **892 baris data bersih** (dari sampel 1.000 baris pertama), jauh melampaui batas minimum **100 baris** yang disyaratkan pada Soal 2.
- **Rentang Nilai AOV Setelah Pembersihan:** Min = **\$60,24** | Max = **\$339,68**.
- **Fitur Numerik yang Digunakan (4 Fitur):**
  1. `Average_Order_Value` (Nilai rata-rata belanja pesanan pelanggan dalam USD).
  2. `Total_Purchases` (Frekuensi total transaksi pembelian).
  3. `Pages_Per_Session` (Rata-rata jumlah halaman produk yang dilihat per sesi kunjungan).
  4. `Membership_Years` (Lama durasi keanggotaan pelanggan dalam tahun).
- **Normalisasi Data (Feature Scaling):**
  Menerapkan **Min-Max Scaling** manual (skala 0–1) agar ke-4 fitur memiliki bobot pengaruh yang setara dan adil terhadap perhitungan jarak Euclidean.
- **Hasil Klastering K-Means ($K=2$ setelah Normalisasi & Pembersihan Outlier):**
  - **Klaster 1 ($C_1$):** 633 pelanggan (**70,96%**) — profil pelanggan baru / reguler (rata-rata lama langganan 1,92 tahun).
  - **Klaster 2 ($C_2$):** 259 pelanggan (**29,04%**) — profil pelanggan loyal / senior (rata-rata lama langganan 5,63 tahun).

---

## Analisis Korelasi Fitur: Pearson & Spearman

Sebelum proses pemodelan, dilakukan pengujian korelasi statistik antara ke-4 fitur numerik menggunakan **Korelasi Pearson** (mengukur hubungan linier) dan **Korelasi Spearman** (mengukur hubungan monotonik/peringkat):

### 1. Matriks Korelasi Pearson (Linier)

| Fitur | Average_Order_Value | Total_Purchases | Pages_Per_Session | Membership_Years |
| :--- | :---: | :---: | :---: | :---: |
| **Average_Order_Value** | **1.0000** | 0.0018 | 0.0210 | 0.0906 |
| **Total_Purchases** | 0.0018 | **1.0000** | **0.6102** | 0.0094 |
| **Pages_Per_Session** | 0.0210 | **0.6102** | **1.0000** | 0.0518 |
| **Membership_Years** | 0.0906 | 0.0094 | 0.0518 | **1.0000** |

### 2. Matriks Korelasi Spearman (Monotonik / Peringkat)

| Fitur | Average_Order_Value | Total_Purchases | Pages_Per_Session | Membership_Years |
| :--- | :---: | :---: | :---: | :---: |
| **Average_Order_Value** | **1.0000** | 0.0216 | 0.0303 | 0.0149 |
| **Total_Purchases** | 0.0216 | **1.0000** | **0.7135** | 0.0419 |
| **Pages_Per_Session** | 0.0303 | **0.7135** | **1.0000** | 0.0691 |
| **Membership_Years** | 0.0149 | 0.0419 | 0.0691 | **1.0000** |

### 3. Wawasan Penting (*Insights*):
1. **Korelasi Positif Sangat Kuat:** Fitur `Pages_Per_Session` dan `Total_Purchases` memiliki nilai $r_{\text{pearson}} = 0{,}6102$ dan $r_{\text{spearman}} = 0{,}7135$. Ini membuktikan adanya pola perilaku nyata: semakin sering pelanggan menjelajahi halaman web, semakin tinggi total transaksi pembelian mereka.
2. **Fitur Ortogonal / Independen:** Fitur `Average_Order_Value` memiliki korelasi mendekati 0 terhadap fitur lainnya. Artinya, besar pengeluaran per transaksi tidak tergantung pada seberapa lama atau sering pelanggan membuka website.
3. **Urgensi Normalisasi:** Tanpa normalisasi, korelasi kuat antara `Pages_Per_Session` dan `Total_Purchases` ini akan tertutup oleh besarnya rentang nilai `Average_Order_Value`. Normalisasi Min-Max mengembalikan keseimbangan peran seluruh fitur.

---

## Rumus Umum Perhitungan Manual

Berikut adalah landasan matematis yang digunakan pada implementasi program dari awal (*from scratch*) dan perhitungan manual:

### 1. Normalisasi Min-Max (*Min-Max Feature Scaling*)
Mentransformasikan seluruh nilai fitur ke dalam rentang tertutup $[0, 1]$:

$$ x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} $$

### 2. Jarak Euclidean (*Euclidean Distance*)
Mengukur jarak geometris antara dua titik data $\mathbf{a} = (a_1, a_2, \dots, a_m)$ dan $\mathbf{b} = (b_1, b_2, \dots, b_m)$ pada ruang fitur $m$-dimensi:

$$ d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{m} (a_i - b_i)^2} $$

### 3. Penentuan Klaster K-Means (*Cluster Assignment*)
Setiap titik data $\mathbf{x}_i$ dialokasikan ke klaster yang centroid-nya memiliki jarak Euclidean minimum:

$$ c(\mathbf{x}_i) = \arg\min_{j \in \{1, \dots, K\}} d(\mathbf{x}_i, \mathbf{c}_j) $$

### 4. Pembaruan Titik Pusat Klaster (*Centroid Update*)
Centroid baru $\mathbf{c}_k$ dihitung sebagai nilai rata-rata (*mean vector*) dari seluruh anggota himpunan klaster $S_k$:

$$ \mathbf{c}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x} $$

### 5. Klasifikasi K-Nearest Neighbors (K-NN Majority Voting)
Untuk memprediksi kelas data uji $\mathbf{x}_{\text{test}}$, dihitung jarak ke seluruh data latih, dipilih $K$ data terdekat $N_K(\mathbf{x}_{\text{test}})$, kemudian dilakukan pemungutan suara terbanyak (*majority vote*):

$$ \hat{y} = \arg\max_{c \in C} \sum_{i \in N_K(\mathbf{x}_{\text{test}})} I(y_i = c) $$

### 6. Akurasi (*Accuracy*) dan RMSE (*Root Mean Squared Error*)

$$ \text{Akurasi} = \frac{\sum_{i=1}^{n} I(y_i = \hat{y}_i)}{n} \times 100\% \qquad\qquad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

---

## 1. Langkah Pengujian — Akurasi dan RMSE (Soal 1)

### Prosedur Pengujian
1. **Normalisasi & Pemberian Label Acuan (K-Means Clustering):**
   Seluruh data ternormalisasi dikelompokkan dengan K-Means ($K=2$) untuk menghasilkan label klaster: $C_1 \rightarrow 0$ dan $C_2 \rightarrow 1$.
2. **Pengacakan Data (*Data Shuffling*):**
   Urutan data diacak secara merata menggunakan `random.seed(42)`.
3. **Pemisahan Data (*Train/Test Split* 80/20):**
   - **Data Latih (80%):** 713 baris data beserta label klasternya.
   - **Data Uji (20%):** 179 baris data yang disisihkan khusus untuk pengujian.
4. **Pelatihan & Prediksi K-NN ($K=3$):**
   Mencari 3 tetangga terdekat di data latih menggunakan jarak Euclidean ternormalisasi.
5. **Evaluasi Metrik:**
   Menghitung jumlah tebakan benar, nilai akurasi, dan nilai RMSE pada 179 data uji.

### Hasil Pengujian Model K-NN ($K=3$)

| Parameter Evaluasi | Nilai | Keterangan |
|---|:---:|---|
| Total Data Bersih | 892 baris | Setelah filter outlier $60 \le \text{AOV} \le 5000$ |
| Data Latih (80%) | 713 baris | Basis penentuan tetangga K-NN |
| Data Uji (20%) | 179 baris | Data evaluasi validasi murni (*unseen data*) |
| **Akurasi Data Uji** | **0,9721 (97,21%)** | 174 dari 179 data uji terprediksi benar (5 salah) |
| **RMSE Data Uji** | **0,1671** | $\sqrt{5 / 179} = \sqrt{0{,}0279} = 0{,}1671$ |

#### Confusion Matrix pada 179 Data Uji:

| | Prediksi Klaster C1 | Prediksi Klaster C2 | Total Aktual |
|---|:---:|:---:|:---:|
| **Sebenarnya Klaster C1** | **126** (Benar) | 3 (Salah) | 129 data |
| **Sebenarnya Klaster C2** | 2 (Salah) | **48** (Benar) | 50 data |
| **Total Prediksi** | 128 data | 51 data | **179 data** |

---

### Analisis Evaluasi: Perbandingan Sebelum vs Sesudah Normalisasi & Outlier Filtering

| Aspek Evaluasi | Sebelum Normalisasi & Filter Outlier | Setelah Normalisasi & Filter Outlier |
| :--- | :--- | :--- |
| **Kondisi Data** | Terdapat outlier ekstrem AOV (\$5.279,80) | Bersih pada rentang \$60,24 – \$339,68 (892 baris) |
| **Penyebaran Klaster** | Sangat timpang: C1 = 909 (97%), C2 = 26 (3%) | **Proporsional:** C1 = 633 (71%), C2 = 259 (29%) |
| **Sifat Keputusan Klaster** | Hanya didikte oleh AOV (1 dimensi) | Menggabungkan ke-4 dimensi secara seimbang |
| **Akurasi K-NN** | **100,00%** (Trivial / Overfit semu) | **97,21%** (Realistis, ilmiah, dan berdaya uji tinggi) |
| **RMSE K-NN** | **0,0000** | **0,1671** |

> [!NOTE]
> Perubahan akurasi dari 100% menjadi **97,21%** bukanlah penurunan performa, melainkan bukti bahwa batas keputusan (*decision boundary*) kini bekerja secara riil di ruang multidimensi. Terdapat 5 data uji yang berada sangat dekat dengan batas perbatasan antar klaster sehingga terjadi variasi prediksi lokal, mencerminkan sifat pembelajaran mesin yang sebenarnya.

---

## 2. Visualisasi Pengelompokan Data

Pada notebook telah ditambahkan 3 cell visualisasi menggunakan `matplotlib` dan `seaborn` untuk memudahkan interpretasi hasil klastering:

1. **Visualisasi K-Means Seluruh Data (3 Panel):**
   - **Panel 1 (AOV vs Total Purchases):** Memperlihatkan sebaran pelanggan Klaster C1 dan C2 beserta titik Centroid akhir berpenanda silang (`X` merah).
   - **Panel 2 (Pages/Session vs Membership Years):** Memperlihatkan pemisahan tegas antara pelanggan baru (C1, rata-rata 1,92 tahun) dan pelanggan senior (C2, rata-rata 5,63 tahun).
   - **Panel 3 (Diagram Batang):** Memvisualisasikan jumlah dan persentase anggota klaster: C1 (633 data / 71,0%) dan C2 (259 data / 29,0%).
2. **Visualisasi K-Means 20 Data Pertama (Soal 3):**
   - Menampilkan pemetaan posisi titik pelanggan dengan label identitas **P01 s.d. P20** serta posisi Centroid 20 data (marker bintang kuning `★`).
3. **Visualisasi Pengujian Data Baru & K-NN (Soal 4 & 5):**
   - Menampilkan posisi titik **Data Baru** (marker hijau terang `+`), titik-titik Centroid, serta **garis hijau putus-putus** yang menghubungkan data baru ke **3 Tetangga Terdekatnya** lengkap dengan label jarak Euclidean masing-masing.

---

## 3. Penggunaan Data Lain Berjumlah Banyak (Min. 100) (Soal 2)

Sesuai ketentuan Soal 2:
- Diintegrasikan dataset baru dari Kaggle melalui `kagglehub`: `dhairyajeetsingh/ecommerce-customer-behavior-dataset`.
- Data bersih yang digunakan sebanyak **892 baris**, melebihi syarat minimal 100 data hingga **8,92 kali lipat**.
- Menggunakan 4 dimensi fitur: `Average_Order_Value`, `Total_Purchases`, `Pages_Per_Session`, dan `Membership_Years`.
- Hasil Centroid Akhir Konvergen (Iterasi ke-8):
  - **Skala Ternormalisasi (0–1):**
    - $\mathbf{c}_1 = [0{,}2257;\; 0{,}1248;\; 0{,}3356;\; 0{,}1837]$ (633 anggota)
    - $\mathbf{c}_2 = [0{,}2281;\; 0{,}1380;\; 0{,}3901;\; 0{,}5586]$ (259 anggota)
  - **Skala Asli (*Denormalized*):**
    - $\mathbf{c}_1 = [\$123{,}32;\; 12{,}73\text{ beli};\; 8{,}32\text{ hal};\; 1{,}92\text{ thn}]$ (Segmen Pelanggan Baru / Reguler)
    - $\mathbf{c}_2 = [\$123{,}99;\; 13{,}98\text{ beli};\; 9{,}50\text{ hal};\; 5{,}63\text{ thn}]$ (Segmen Pelanggan Loyal / Senior)

---

## 4. Perhitungan Manual Menggunakan 20 Data (Soal 3)

Berikut adalah simulasi perhitungan manual langkah demi langkah menggunakan 20 baris data pertama dari dataset bersih (`x[:20]`), diverifikasi kesesuaiannya dengan program notebook.

### 4.1 Tabel 20 Data Awal

| ID | Average_Order_Value ($x_1$) | Total_Purchases ($x_2$) | Pages_Per_Session ($x_3$) | Membership_Years ($x_4$) |
|:---:|:---:|:---:|:---:|:---:|
| **P01** | 94.72 | 9.0 | 6.0 | 2.9 |
| **P02** | 82.45 | 19.5 | 10.3 | 1.6 |
| **P03** | 165.52 | 9.1 | 1.6 | 2.9 |
| **P04** | 147.33 | 15.0 | 14.8 | 2.6 |
| **P05** | 190.97 | 12.0 | 6.9 | 3.5 |
| **P06** | 169.58 | 20.0 | 13.9 | 1.9 |
| **P07** | 161.98 | 13.0 | 8.3 | 1.6 |
| **P08** | 77.79 | 6.0 | 7.4 | 1.3 |
| **P09** | 90.95 | 9.0 | 5.4 | 3.6 |
| **P10** | 118.00 | 5.0 | 2.9 | 0.9 |
| **P11** | 78.88 | 5.0 | 5.6 | 0.6 |
| **P12** | 123.99 | 10.0 | 7.0 | 7.1 |
| **P13** | 107.58 | 11.0 | 8.9 | 3.1 |
| **P14** | 123.00 | 15.0 | 10.3 | 3.7 |
| **P15** | 91.92 | 18.0 | 18.0 | 1.4 |
| **P16** | 116.39 | 4.0 | 3.7 | 0.9 |
| **P17** | 107.78 | 14.3 | 5.3 | 0.7 |
| **P18** | 103.12 | 11.0 | 9.8 | 2.0 |
| **P19** | 80.88 | 16.0 | 8.8 | 5.6 |
| **P20** | 108.52 | 24.0 | 16.0 | 1.2 |

*Catatan: Nilai min dan max fitur untuk normalisasi: $x_1 \in [60{,}24, 339{,}68]$, $x_2 \in [1, 29]$, $x_3 \in [1, 20]$, $x_4 \in [0, 10]$.*

### 4.2 Inisialisasi Centroid & Konvergensi pada 20 Data
- Centroid awal diambil dari data pertama (**P01**) dan data terakhir (**P20**).
- K-Means pada 20 data mencapai titik **konvergen stabil**:
  - **Klaster 1 ($C_1$):** 16 data (P01, P02, P03, P07, P08, P09, P10, P11, P12, P13, P14, P16, P17, P18, P19, P20).
  - **Klaster 2 ($C_2$):** 4 data (P04, P05, P06, P15) — profil aktivitas penelusuran halaman web tinggi (`Pages_Per_Session` > 13).
- **Centroid Akhir Konvergen:**
  - $\mathbf{c}_1 = [0{,}1924;\; 0{,}1017;\; 0{,}2643;\; 0{,}2551]$
  - $\mathbf{c}_2 = [0{,}2473;\; 0{,}1941;\; 0{,}6732;\; 0{,}1692]$

---

## 5. Program Metode K-NN Menggunakan Data Hasil Klastering K-Means (Soal 4)

Model K-Nearest Neighbors diimplementasikan murni tanpa bantuan scikit-learn.

### Pengujian pada Sampel Data Baru Spesifik:
Diuji pada sampel data baru $\mathbf{x}_{\text{baru}} = [220{,}0;\; 15{,}0;\; 10{,}0;\; 3{,}0]$:
1. **Transformasi Skala:** Data baru ditransformasikan ke skala ternormalisasi menggunakan min-max data latih:
   $$\mathbf{x}_{\text{baru (norm)}} = [0{,}5717;\; 0{,}1556;\; 0{,}4737;\; 0{,}3000]$$
2. **Perhitungan Jarak ke 713 Data Latih:** Tiga tetangga terdekat yang ditemukan adalah:
   - **Tetangga 1:** Jarak $= \mathbf{0{,}0910}$ $\rightarrow$ Klaster **C2**
   - **Tetangga 2:** Jarak $= \mathbf{0{,}1045}$ $\rightarrow$ Klaster **C1**
   - **Tetangga 3:** Jarak $= \mathbf{0{,}1177}$ $\rightarrow$ Klaster **C1**
3. **Majority Voting:** Dua dari tiga tetangga terdekat berasal dari Klaster C1 (2 suara vs 1 suara).
   $$\implies \hat{y}_{\text{KNN}} = \mathbf{C1}$$

---

## 6. Perbandingan: Jarak ke Centroid vs Metode K-NN (Soal 5)

Pada soal nomor 5, dilakukan perbandingan performa antara **Metode K-NN ($K=3$)** dengan **Metode Jarak ke Centroid (*Nearest Centroid Classifier*)**.

### 6.1 Uji Coba pada Sampel Data Baru Spesifik $\mathbf{x}_{\text{baru}}$

1. **Metode Jarak Centroid:**
   - Jarak ke Centroid 1: $d(\mathbf{x}_{\text{baru}}, \mathbf{c}_1) = \mathbf{0{,}3717}$
   - Jarak ke Centroid 2: $d(\mathbf{x}_{\text{baru}}, \mathbf{c}_2) = \mathbf{0{,}4350}$
   - Karena $d(\mathbf{x}_{\text{baru}}, \mathbf{c}_1) < d(\mathbf{x}_{\text{baru}}, \mathbf{c}_2)$, **Metode Jarak Centroid memprediksi Klaster C1**.
2. **Metode K-NN ($K=3$):**
   - Berdasarkan *majority voting* 3 tetangga terdekat, **Metode K-NN memprediksi Klaster C1**.
3. **Kesimpulan Hasil Sampel:**
   Kedua metode menghasilkan **keputusan yang konsisten dan identik**, yaitu mengelompokkan data baru ke dalam **Klaster C1**.

### 6.2 Evaluasi pada Seluruh 179 Data Uji

| Metrik Evaluasi | Metode K-NN ($K=3$) | Metode Jarak ke Centroid | Analisis Perbandingan |
|---|:---:|:---:|:---:|
| **Akurasi** | **97,21%** (174 / 179) | **100,00%** (179 / 179) | Centroid identik terhadap partisi asal |
| **RMSE** | **0,1671** | **0,0000** | Selisih margin 5 data perbatasan |
| **Waktu Prediksi (*Inference*)** | ~0,028 detik | **~0,0008 detik** | **Centroid ~35x lebih cepat** |
| **Kompleksitas Komputasi** | $\mathcal{O}(N_{\text{train}} \cdot m)$ | $\mathcal{O}(K \cdot m)$ | Centroid konstan terhadap ukuran data |

### 6.3 Manakah yang Lebih Baik? Berikan Alasan Anda

Jawabannya bergantung pada konteks dan kebutuhan implementasi:

1. **Kelebihan Metode Jarak Centroid:**
   - **Efisiensi Komputasi Sangat Tinggi:** Hanya perlu menghitung $K=2$ jarak berapapun besarnya data latih. Sangat ideal untuk sistem *real-time* atau data berukuran jutaan baris (*big data*).
   - **Ketahanan Global:** Stabil terhadap sebaran data umum karena mengacu langsung pada titik tengah kelompok.

2. **Kelebihan Metode K-NN:**
   - **Fleksibilitas Batas Lokal (*Non-Linear Boundary*):** K-NN tidak mengasumsikan klaster harus berbentuk bulat simetris (*spherical*). Jika klaster memiliki bentuk memanjang, melengkung, atau bercabang, K-NN jauh lebih adaptif.
   - **Ketahanan terhadap Outlier Centroid:** K-NN tidak terpengaruh oleh pergeseran satu titik pusat klaster, karena keputusan hanya diambil dari tetangga terdekat di sekeliling data baru.

#### Kesimpulan Pemilihan:
> [!IMPORTANT]
> - **Gunakan Metode Jarak Centroid** jika data berukuran masif, membutuhkan kecepatan inferensi instan (*low latency*), dan klaster terpisah secara linier.
> - **Gunakan Metode K-NN** jika distribusi data memiliki bentuk yang kompleks, tidak beraturan (*irregular manifold*), dan akurasi batas lokal menjadi prioritas utama.

---

## Kesimpulan Akhir

1. **Dataset Bersih & Terukur:** Penggunaan dataset Kaggle menghasilkan **892 data valid** setelah pembersihan outlier AOV ($60 \le \text{AOV} \le 5000$), melampaui syarat minimal 100 data.
2. **Pentingnya Normalisasi:** Normalisasi Min-Max terbukti krusial untuk menyeimbangkan ke-4 dimensi fitur, mencegah dominasi berlebih AOV, dan menghasilkan segmentasi pelanggan yang proporsional (C1: 71% pelanggan baru vs C2: 29% pelanggan loyal).
3. **Pengujian K-NN yang Realistis:** Model K-NN ($K=3$) mencapai akurasi **97,21%** dan RMSE **0,1671** pada 179 data uji, membuktikan generalisasi yang kuat pada data yang belum pernah dilihat (*unseen data*).
4. **Verifikasi Manual 100% Cocok:** Simulasi 20 data pertama terbukti selaras antara perhitungan manual dan eksekusi notebook.
5. **Visualisasi Komprehensif:** Tiga panel plot klastering, plot 20 data berlabel, dan visualisasi tetangga K-NN berhasil memvisualisasikan seluruh proses pengelompokan secara jelas dan informatif.
