# Laporan 2 — Naive Bayes (Klasifikasi SMS Spam/Ham)

Notebook: `notebook/NaiveBayes.ipynb`

## Dataset

Dataset diambil dari Kaggle lewat API (`kagglehub`):

```python
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
```

- Sumber: **SMS Spam Collection** (UCI), file `spam.csv` (~211 KB).
- Jumlah data: **5.572 baris** (jauh melebihi minimum 100).
- Kolom asli `v1` = label (`ham`/`spam`), `v2` = isi pesan teks; kolom `Unnamed: 2-4` gugur karena diseleksi `df[["v1","v2"]]`.

Karena model Naive Bayes yang dipakai **kategorikal**, isi pesan teks diekstrak menjadi 4 fitur kategorikal lewat fungsi `ekstrak_fitur` (ada di cell fungsi):

| Fitur | Nilai | Cara hitung |
|---|---|---|
| panjang | pendek / panjang | `pendek` jika jumlah kata ≤ `BATAS_KATA_PENDEK` (=5), selain itu `panjang` |
| angka | ada / tidak | `ada` jika pesan mengandung digit |
| simbol | ada / tidak | `ada` jika pesan mengandung `!$#@%` |
| kata_spam | ada / tidak | `ada` jika pesan mengandung kata *free, win, call, text, claim, prize, urgent, reward* (cocok **per kata**, bukan substring, lewat regex word-boundary) |

Distribusi kelas pada 5.572 data: **ham = 4.825 (86,59%)**, spam = 747 (13,41%).
Distribusi fitur `kata_spam`: ham → tidak 4.483 / ada 342; spam → ada 561 / tidak 186.

---

## Rumus umum perhitungan manual (Naive Bayes)

**1. Prior:**

$$P(k) = \frac{n_k}{N}$$

**2. Likelihood** (asumsi fitur saling bebas, jadi tiap fitur dikalikan):

$$P(\mathbf{x} \mid k) = \prod_{i=1}^{m} P(x_i \mid k), \qquad P(x_i = v \mid k) = \frac{\text{jumlah data kelas } k \text{ dengan } x_i = v}{n_k}$$

**3. Posterior (aturan Bayes, dinormalisasi):**

$$P(k \mid \mathbf{x}) = \frac{P(k)\, P(\mathbf{x} \mid k)}{\sum_{j} P(j)\, P(\mathbf{x} \mid j)}$$

Kelas dengan $P(k\mid\mathbf{x})$ terbesar jadi prediksi. Nilai likelihood yang belum pernah muncul di latih dipakai *smoothing* kecil $10^{-6}$.

**4. Akurasi:** $\text{Accuracy} = \dfrac{\text{jumlah prediksi benar}}{n}\times 100\%$

**5. RMSE** (label encode `ham=0`, `spam=1`):

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

---

## 1. Langkah pengujian — akurasi dan RMSE

### Langkah pengujian

1. Encode label `ham=0`, `spam=1` (urut alfabet) untuk RMSE.
2. Acak urutan data (`random.seed(42)`).
3. Split **80% data latih (4.457)** dan **20% data uji (1.115)**.
4. Bangun model Naive Bayes dari **data latih saja** (`naiveBayes`).
5. Prediksi tiap baris data uji (`prediksi_naiveBayes`).
6. Hitung akurasi dan RMSE pada data uji.

### Hasil

| Metrik | Nilai |
|---|---|
| Data latih | 4.457 |
| Data uji | 1.115 |
| Akurasi data uji | **0,9552 (95,52%)** |
| RMSE data uji | **0,2118** |

Konsistensi: error rate = $1 - 0{,}9552 = 0{,}0448$, maka $\text{RMSE} = \sqrt{0{,}0448} \approx 0{,}2118$ ✓

**Baseline** "tebak semua ham" = 85,83%. Model NB (95,52%) jauh melampaui baseline.

**Confusion matrix NB pada data uji:**

| | Prediksi spam | Prediksi ham |
|---|---|---|
| **Sebenarnya spam** | 118 (benar) | 40 (salah) |
| **Sebenarnya ham** | 10 (salah) | 947 (benar)

Spam terlewat (FN=40) lebih banyak dari ham yang tersangka (FP=10) — wajar karena prior ham jauh lebih besar (0,8659).

---

## 2. Penggunaan data lain dengan jumlah lebih banyak (min. 100)

- Tugas sebelumnya memakai dataset cuaca/play-tennis lokal **20 baris**.
- Data diganti ke **SMS Spam Collection** dengan **5.572 baris** (melebihi minimum 100).
- Pembacaan: `pd.read_csv(data, encoding="latin-1")` — `latin-1` wajib karena file mengandung karakter non-UTF8.

---

## 3. Perhitungan manual dengan 20 data

### 3.1 Data yang dipakai (20 baris pertama)

| # | panjang | angka | simbol | kata_spam | label |
|---|---|---|---|---|---|
| 1 | panjang | tidak | tidak | tidak | ham |
| 2 | panjang | tidak | tidak | tidak | ham |
| 3 | panjang | ada | tidak | ada | spam |
| 4 | panjang | tidak | tidak | tidak | ham |
| 5 | panjang | tidak | tidak | tidak | ham |
| 6 | panjang | ada | ada | tidak | spam |
| 7 | panjang | tidak | tidak | tidak | ham |
| 8 | panjang | ada | tidak | tidak | ham |
| 9 | panjang | ada | ada | ada | spam |
| 10 | panjang | ada | ada | ada | spam |
| 11 | panjang | tidak | tidak | tidak | ham |
| 12 | panjang | ada | ada | ada | spam |
| 13 | panjang | ada | ada | ada | spam |
| 14 | panjang | tidak | tidak | tidak | ham |
| 15 | panjang | tidak | ada | tidak | ham |
| 16 | panjang | tidak | tidak | tidak | spam |
| 17 | pendek | tidak | tidak | tidak | ham |
| 18 | panjang | ada | tidak | tidak | ham |
| 19 | panjang | tidak | tidak | tidak | ham |
| 20 | panjang | ada | tidak | tidak | spam |

Jumlah: **ham = 12, spam = 8, $N = 20$**.

### 3.2 Prior (manual)

$$P(\text{ham}) = \frac{12}{20} = 0{,}6, \qquad P(\text{spam}) = \frac{8}{20} = 0{,}4$$

### 3.3 Likelihood (manual)

**Kelas ham ($n = 12$):**

| Fitur | Nilai | Hitung | $P(x_i\mid\text{ham})$ |
|---|---|---|---|
| panjang | panjang | 11/12 | 0,9167 |
| panjang | pendek | 1/12 | 0,0833 |
| angka | tidak | 10/12 | 0,8333 |
| angka | ada | 2/12 | 0,1667 |
| simbol | tidak | 11/12 | 0,9167 |
| simbol | ada | 1/12 | 0,0833 |
| kata_spam | tidak | 12/12 | 1,0000 |

*(tidak ada data ham dengan kata_spam=ada pada 20 baris ini; untuk kata_spam=ada dipakai smoothing $10^{-6}$.)*

**Kelas spam ($n = 8$):**

| Fitur | Nilai | Hitung | $P(x_i\mid\text{spam})$ |
|---|---|---|---|
| panjang | panjang | 8/8 | 1,0000 |
| angka | ada | 7/8 | 0,8750 |
| angka | tidak | 1/8 | 0,1250 |
| simbol | tidak | 3/8 | 0,3750 |
| simbol | ada | 5/8 | 0,6250 |
| kata_spam | ada | 5/8 | 0,6250 |
| kata_spam | tidak | 3/8 | 0,3750 |

### 3.4 Perhitungan posterior manual (contoh 4 data)

**Data #1** — $\mathbf{x} = [\text{panjang}, \text{tidak}, \text{tidak}, \text{tidak}]$, asli ham:

$$P(\text{ham})P(\mathbf{x}\mid\text{ham}) = 0{,}6 \times 0{,}9167 \times 0{,}8333 \times 0{,}9167 \times 1{,}0 = 0{,}420139$$
$$P(\text{spam})P(\mathbf{x}\mid\text{spam}) = 0{,}4 \times 1{,}0 \times 0{,}1250 \times 0{,}3750 \times 0{,}3750 = 0{,}007031$$
$$\text{Total} = 0{,}427170 \Rightarrow P(\text{ham}\mid\mathbf{x}) = 0{,}9835,\; P(\text{spam}\mid\mathbf{x}) = 0{,}0165 \Rightarrow \textbf{ham (benar)}$$

**Data #6** — $\mathbf{x} = [\text{panjang}, \text{ada}, \text{ada}, \text{tidak}]$, asli spam:

$$P(\text{ham}\mid\mathbf{x}) \propto 0{,}6 \times 0{,}9167 \times 0{,}1667 \times 0{,}0833 \times 1{,}0 = 0{,}007639$$
$$P(\text{spam}\mid\mathbf{x}) \propto 0{,}4 \times 1{,}0 \times 0{,}8750 \times 0{,}6250 \times 0{,}3750 = 0{,}082031$$
$$\Rightarrow P(\text{ham}\mid\mathbf{x}) = 0{,}0852,\; P(\text{spam}\mid\mathbf{x}) = 0{,}9148 \Rightarrow \textbf{spam (benar)}$$

**Data #16** — $\mathbf{x} = [\text{panjang}, \text{tidak}, \text{tidak}, \text{tidak}]$, asli spam:

$$P(\text{ham}\mid\mathbf{x}) = 0{,}9835,\; P(\text{spam}\mid\mathbf{x}) = 0{,}0165 \Rightarrow \textbf{ham (SALAH)}$$

Pesan spam ini tidak mengandung angka/simbol/kata-spam, jadi model mengira ham.

**Data #20** — $\mathbf{x} = [\text{panjang}, \text{ada}, \text{tidak}, \text{tidak}]$, asli spam:

$$P(\text{ham}\mid\mathbf{x}) \propto 0{,}6 \times 0{,}9167 \times 0{,}1667 \times 0{,}9167 \times 1{,}0 = 0{,}084028$$
$$P(\text{spam}\mid\mathbf{x}) \propto 0{,}4 \times 1{,}0 \times 0{,}8750 \times 0{,}3750 \times 0{,}3750 = 0{,}049219$$
$$\Rightarrow P(\text{ham}\mid\mathbf{x}) = 0{,}6306,\; P(\text{spam}\mid\mathbf{x}) = 0{,}3694 \Rightarrow \textbf{ham (SALAH)}$$

### 3.5 Pengujian ulang 20 data (resubstitution)

Salah prediksi pada **data #16 dan #20** (2 dari 20):

$$\text{Akurasi} = \frac{18}{20} = 0{,}9000\;(90\%), \qquad \text{RMSE} = \sqrt{\frac{2}{20}} = \sqrt{0{,}1} = \mathbf{0{,}3162}$$

---

## 4. Perbandingan dengan Decision Tree (data 20 yang sama)

Fungsi DT (`entropy`, `gain`, `buat_tree`) dari laporan 1 dijalankan pada **20 data & fitur yang sama** (cell `banding-decision-tree`).

### Peringkat gain (DT pada 20 data)

Entropy awal:

$$H(S) = -\frac{12}{20}\log_2\frac{12}{20} - \frac{8}{20}\log_2\frac{8}{20} = 0{,}9710$$

| Fitur | Gain |
|---|---|
| **kata_spam** | **0,4295** ← akar |
| angka | 0,3853 |
| simbol | 0,2512 |
| panjang | 0,0381 |

### Pohon hasil DT (20 data)

```
                      kata_spam
                 ┌────────┴────────┐
               ada               tidak
                │                  └── angka
             spam (murni)        ┌─────┴─────┐
                              ada         tidak
                               │            └── simbol
                            simbol        ┌─────┴─────┐
                          ┌────┴────┐   tidak      ada
                        tidak      ada    │          └── spam
                          │         └── spam
                       panjang
                     ┌────┴────┐
                   pendek    panjang
                     │          └── ham
                    ham
```

### Hasil DT pada 20 data

Salah prediksi pada **data #16 dan #20** (sama dengan NB):

$$\text{Akurasi} = \frac{18}{20} = 0{,}9000\;(90\%), \qquad \text{RMSE} = \mathbf{0{,}3162}$$

### Perbandingan langsung (20 data)

| Metrik | Naive Bayes | Decision Tree |
|---|---|---|
| Akurasi | 90,00% | 90,00% |
| RMSE | 0,3162 | 0,3162 |
| Data salah | #16, #20 | #16, #20 |

Pada 20 data ini keduanya **benar-benar identik** — sama-sama prediksi, sama-sama 2 salah (#16 & #20). Karena fitur `kata_spam` mendominasi, NB dan DT sama-sama menjadikannya penentu utama.

---

## 5. Perbandingan besar: Naive Bayes vs Decision Tree (data uji 1.115)

Kedua model diuji pada **data uji 1.115 yang sama** (split 80/20 seed 42, model dari 4.457 data latih).

| Metrik | Naive Bayes | Decision Tree |
|---|---|---|
| Akurasi | **95,52%** | **95,52%** |
| RMSE | **0,2118** | **0,2118** |
| Spam benar (TP) | 118 | 118 |
| Spam terlewat (FN) | 40 | 40 |
| Ham salah tersangka (FP) | 10 | 10 |
| Ham benar (TN) | 947 | 947 |

Keduanya menghasilkan **nilai identik** pada data uji besar.

**Mengapa identik?** Fitur hanya 4, nilainya kategorikal/biner, dan jumlah kombinasi fitur terbatas. Baik NB (probabilistik, asumsi fitur bebas) maupun DT (memecah berdasarkan gain) pada akhirnya belajar **batas keputusan yang sama**. Perbedaan baru terasa bila fitur sangat banyak (NB unggul), fitur punya interaksi kuat (DT unggul), atau data penuh/noisy.

---

## 6. Mana yang lebih baik — Naive Bayes atau Decision Tree?

**Untuk dataset ini keduanya setara (95,52% / 0,2118, identik).** Namun secara umum:

**Pilih Naive Bayes jika:**
- Fitur **banyak** dan sebagian tidak relevan — NB tahan fitur noise (tiap fitur dinilai terpisah).
- Butuh **latih cepat** & model ringan — NB hanya menghitung kemunculan; DT memecah rekursif.
- Data teks (spam, sentiment) — NB lazim dipakai.

**Pilih Decision Tree jika:**
- Butuh **aturan mudah dibaca** ("jika kata_spam=ada → spam") — DT menghasilkan pohon eksplisit.
- Fitur punya **interaksi** kuat yang tak tertangani asumsi bebas NB.
- Target non-linear tanpa log probabilitas.

**Kesimpulan tugas ini:** pada 4 fitur ekstraksi sederhana, keduanya berperforma sama; tidak ada pemenang mutlak. NB unggul pada kesederhanaan probabilistik + posterior (mis. 98,35% ham), DT unggul pada interpretabilitas aturan. Untuk skala teks nyata dengan banyak fitur → **Naive Bayes lebih praktis**; untuk keputusan yang perlu dijelaskan ke non-teknis → **Decision Tree lebih transparan**.

---

## Kesimpulan

1. Pengujian 80/20 pada 5.572 data SMS: **akurasi 95,52%, RMSE 0,2118** — jauh di atas baseline ham-saja (85,83%).
2. Data dinaikkan dari 20 baris (cuaca) menjadi **5.572 baris** dataset SMS spam via API Kaggle (`kagglehub`).
3. Perhitungan manual 20 data (prior ham 0,6 / spam 0,4, likelihood per fitur, posterior contoh data #1/#6/#16/#20) → akurasi resubstitution **90%, RMSE 0,3162** (salah di #16 & #20).
4. Decision Tree pada 20 data yang sama juga **90% / 0,3162**, salah di data yang sama (#16 & #20).
5. Pada data uji besar NB dan DT **identik (95,52% / 0,2118)**; pilihan metode bergantung kebutuhan (kecepatan & teks → NB; interpretabilitas & interaksi fitur → DT).
