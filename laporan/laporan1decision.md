# Laporan 1 — Decision Tree (ID3 / C4.5 sederhana)

Notebook: `notebook/desiciontree.ipynb`

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

## 2. Penggunaan data lain dengan jumlah lebih banyak (min. 100)

- Tugas praktikum sebelumnya memakai dataset cuaca/play-tennis lokal dengan **20 baris**.
- Pada laporan ini data diganti ke dataset Kaggle lain (**eCommerce Customer Service Satisfaction**) dan jumlahnya dinaikkan menjadi **10.000 baris** (jauh melebihi minimum 100).
- Pembacaan sampling: `pd.read_csv(data, nrows=10000)`.
- Dengan data sebesar ini, pohon hasil `buat_tree` memiliki **352 simpul daun** — jauh lebih kompleks dibanding pohon 20 data (5 daun), dan waktu training tetap ~0,1 detik.

---

## 3. Perhitungan manual dengan 20 data

Soal 3 dikerjakan **secara manual di laporan ini** (tanpa cell program terpisah), memakai 20 baris pertama dari data yang sama. Semua hasil di bawah sudah diverifikasi dengan perhitungan program `entropy()` dan `gain()`.

### 3.1 Data yang dipakai (20 baris pertama)

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

### 3.2 Entropy awal (manual)

$$H(S) = -\frac{18}{20}\log_2\frac{18}{20} - \frac{2}{20}\log_2\frac{2}{20}$$
$$= -(0,9)\log_2(0,9) - (0,1)\log_2(0,1)$$
$$= -(0,9)(-0,1520) - (0,1)(-3,3219) = 0,1368 + 0,3322 = \mathbf{0,4690}$$

### 3.3 Information gain tiap fitur (manual)

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

### 3.4 Pohon hasil (20 data)

Gain terbesar = `category` (0,2690) → jadi akar pohon. Subset `Order Related` (4 data, label campur) masih dicabangi `Agent Shift`. Di subset itu gain `Agent Shift` = 1,0 (split sempurna): Evening→[1,1], Morning→[0], Split→[0]. Subset lain sudah murni → leaf.

```
                        category
       ┌──────────┬─────────┼──────────┬──────────────┐
  Cancellation  Order Rel.  Product Q  Returns   Shopzilla
       1(puas)      │           1        1          1
                    └─ Agent Shift
                    ├─ Evening  → 1 (puas)
                    ├─ Morning  → 0 (tidak puas)
                    └─ Split    → 0 (tidak puas)
```

Notasi Python: `{category: {2: 1, 6: {shift: {2: 1, 3: 0, 5: 0}}, 9: 1, 11: 1, 12: 1}}`.

### 3.5 Pengujian ulang pada 20 data (resubstitution)

Semua 20 data dilatih lalu diprediksi ulang dengan pohon di atas (telusur sesuai rumus no. 3).

| Data ke- | Sebenarnya | Jalur prediksi | Prediksi |
|---|---|---|---|
| 12 | 0 | category Order Related → shift Morning | 0 ✓ |
| 18 | 0 | category Order Related → shift Morning | 0 ✓ |
| 1–11, 13–17, 19–20 | 1 | cabang murni "1" (atau Order Related → Evening) | 1 ✓ |

- Jumlah benar: 20/20

$$\text{Akurasi} = \frac{20}{20} = 1{,}0000 \;(100\%)$$

$$\text{RMSE} = \sqrt{\frac{1}{20}\sum_{i=1}^{20}(y_i - \hat{y}_i)^2} = \sqrt{\frac{0}{20}} = \mathbf{0,0000}$$

### 3.6 Analisis hasil (20 data vs 10.000 data)

1. **Overfitting pada 20 data.** Akurasi 100% & RMSE 0 terlihat sempurna, tapi itu diukur pada data yang sama dipakai latih (resubstitution). Pohon "hafal" data — contoh: semua Returns diprediksi puas hanya karena pada 20 baris itu belum ada yang komplain. Generalisasi ke data baru tidak terjamin.
2. **Data tidak seimbang.** 18/20 (90%) berlabel puas. Pohon yang selalu menjawab "puas" saja sudah dapat akurasi 90% pada data ini, jadi metrik akurasi menipu; perlu dilihat juga RMSE dan kasus kelas minoritas.
3. **Dengan 10.000 data** pohon menjadi jauh lebih besar (352 daun) tetapi diuji dengan split 80/20: akurasi uji **82,05%** dan RMSE **0,4237** — angka yang jujur karena data uji belum pernah dilihat pohon.
4. **Kesimpulan perbandingan:** makin banyak data, akurasi *resubstitution* turun (dari 100% di 20 data menjadi 82,05% di data uji 2.000), namun hasilnya lebih dipercaya. Inilah gunanya langkah pengujian train/test pada soal no. 1.

---

## Kesimpulan

1. Decision tree ID3 dari scratch berhasil dilatih pada data Kaggle 10.000 baris; hasil pengujian 80/20: **akurasi 82,05%, RMSE 0,4237** (sedikit lebih baik dari baseline 81,19%).
2. Data dinaikkan dari 20 baris (cuaca) menjadi **10.000 baris** dataset layanan pelanggan e-commerce via API Kaggle (`kagglehub`).
3. Perhitungan manual pada 20 data dengan rumus entropy & information gain (entropy 0,4690; gain `category` 0,2690 sebagai akar) menghasilkan pohon kecil yang **overfit** (akurasi resubstitution 100%, RMSE 0) — menunjukkan pentingnya evaluasi pada data terpisah.
