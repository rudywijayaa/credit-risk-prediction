# Laporan Proyek Machine Learning - Rudy Wijaya

## Domain Proyek

Industri keuangan, khususnya penyedia layanan kredit dan perbankan, beroperasi dengan mengandalkan kemampuan manajemen dalam mengukur dan memitigasi risiko. Menurut riset dari *Basel Committee on Banking Supervision*, kegagalan lembaga keuangan dalam memprediksi risiko kredit secara akurat merupakan penyebab utama kebangkrutan operasional [1]. Masalah utama yang sering dihadapi adalah tingginya persentase nasabah yang gagal bayar (*default rate*) setelah dana pinjaman dicairkan.

**Mengapa dan bagaimana masalah tersebut harus diselesaikan:**
Masalah ini sangat krusial karena kerugian finansial akibat satu nasabah yang gagal bayar jauh lebih besar dibandingkan akumulasi margin keuntungan dari beberapa nasabah yang lancar. Oleh karena itu, masalah ini harus diselesaikan dengan mengimplementasikan sistem *Credit Scoring* otomatis berbasis *Machine Learning*. Sistem ini akan menganalisis riwayat data, profil, dan rasio finansial calon nasabah secara komprehensif untuk memprediksi probabilitas gagal bayar secara *real-time* sebelum aplikasi pinjaman disetujui, sehingga keputusan bisnis menjadi lebih objektif dan meminimalkan risiko kerugian.

**Referensi:**
[1] J. Dembiermont, M. Scatigna, and R. Shin, "Global and domestic credit growth," *BIS Quarterly Review*, vol. 14, no. 2, pp. 45-58, 2018.
[2] Home Credit Group, "Home Credit Default Risk Dataset," Kaggle, 2018.

---

## Business Understanding

Proses klarifikasi masalah difokuskan pada penanganan karakteristik data perbankan di dunia nyata, yang umumnya memiliki distribusi target yang sangat tidak seimbang (*highly imbalanced data*), di mana jumlah nasabah lancar selalu jauh mendominasi nasabah bermasalah.

### Problem Statements
* **Pernyataan Masalah 1:** Bagaimana cara membangun model *machine learning* yang mampu mendeteksi calon nasabah berisiko tinggi secara akurat di tengah kondisi data yang sangat timpang (hanya ~8% nasabah yang gagal bayar)?
* **Pernyataan Masalah 2:** Bagaimana menyeimbangkan ambang batas (*threshold*) prediksi model agar dapat meminimalkan risiko kerugian (*False Negative*) tanpa menolak terlalu banyak calon nasabah yang sebenarnya berpotensi menguntungkan bank (*False Positive*)?

### Goals
* **Jawaban Pernyataan Masalah 1:** Mengembangkan model klasifikasi menggunakan algoritma *ensemble learning* berbasis pohon (seperti LightGBM, XGBoost, dan Random Forest) yang dilengkapi dengan teknik penyesuaian bobot kelas dinamis (*class weight adjustment*) untuk mengatasi ketimpangan data.
* **Jawaban Pernyataan Masalah 2:** Melakukan optimasi pada nilai *threshold* probabilitas keputusan model dan mengevaluasinya secara komprehensif menggunakan matriks *Confusion Matrix*, sehingga performa model selaras dengan kebijakan selera risiko (*risk appetite*) bisnis.

### Solution statements
Untuk mencapai tujuan di atas, diusulkan pendekatan menggunakan tiga algoritma yang berbeda sebagai pembanding, dengan satu model utama yang akan dioptimasi secara mendalam:
1.  **Baseline Model 1 (Random Forest):** Menggunakan model Random Forest dengan parameter *bagging* `class_weight='balanced_subsample'` untuk menangani rasio kelas minoritas secara paralel.
2.  **Baseline Model 2 (XGBoost):** Menggunakan *Extreme Gradient Boosting* dengan kalkulasi parameter `scale_pos_weight` otomatis berdasarkan rasio target negatif terhadap positif.
3.  **Model Utama Teroptimasi (LightGBM):** Menggunakan LightGBM sebagai model utama yang akan ditingkatkan performanya (*improvement*) melalui proses **Hyperparameter Tuning menggunakan Optuna**. Model ini juga akan dipasangkan dengan *Custom Threshold Optimization* (disetel pada angka 0.65) untuk mencapai metrik evaluasi bisnis yang paling optimal.

Solusi di atas akan dievaluasi secara terukur menggunakan metrik spesifik yang relevan dengan kasus *imbalanced data*, yakni **ROC-AUC** dan **Recall**.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari kompetisi global **Home Credit Default Risk** yang dipublikasikan di platform Kaggle. Dataset utama (`application_train.csv`) memiliki dimensi yang sangat besar, terdiri dari 307.511 baris observasi dan 122 fitur awal sebelum tahap rekayasa data.

**Tautan Sumber Data:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

### Variabel-variabel pada dataset dikelompokkan sebagai berikut:

**1. Identitas Aplikasi & Target Risiko**
* `SK_ID_CURR`: ID unik digital untuk setiap aplikasi pinjaman calon nasabah.
* `TARGET`: Variabel target biner (1 untuk nasabah dengan gagal bayar, 0 untuk pelunasan lancar).

**2. Karakteristik & Atribut Personal Nasabah**
* `NAME_CONTRACT_TYPE`: Jenis kontrak pinjaman (Cash loans atau Revolving loans).
* `CODE_GENDER`: Jenis kelamin calon nasabah (M / F / XNA).
* `FLAG_OWN_CAR`: Penanda apakah nasabah memiliki mobil pribadi (Y / N).
* `FLAG_OWN_REALTY`: Penanda apakah nasabah memiliki rumah/apartemen (Y / N).
* `CNT_CHILDREN`: Jumlah anak yang dimiliki oleh nasabah.
* `NAME_TYPE_SUITE`: Pendamping nasabah saat mengajukan aplikasi kredit.
* `NAME_INCOME_TYPE`: Kategori sumber pendapatan nasabah (Working, Pensioner, dll).
* `NAME_EDUCATION_TYPE`: Tingkat pendidikan tertinggi yang dicapai nasabah.
* `NAME_FAMILY_STATUS`: Status pernikahan atau keluarga dari nasabah.
* `NAME_HOUSING_TYPE`: Kondisi atau status tempat tinggal nasabah.
* `REGION_POPULATION_RELATIVE`: Skor ternormalisasi populasi wilayah tempat tinggal nasabah.
* `DAYS_BIRTH`: Usia nasabah saat pengajuan (dalam hitungan hari negatif).
* `DAYS_EMPLOYED`: Durasi masa kerja nasabah saat pengajuan (dalam hitungan hari negatif).
* `DAYS_REGISTRATION`: Jeda hari nasabah mengubah registrasi dokumen sebelum pengajuan.
* `DAYS_ID_PUBLISH`: Jeda hari nasabah mengubah identitas lokal sebelum pengajuan.
* `OWN_CAR_AGE`: Usia mobil pribadi milik nasabah (dalam tahun).
* `CNT_FAM_MEMBERS`: Jumlah anggota keluarga nasabah.
* `REGION_RATING_CLIENT` & `REGION_RATING_CLIENT_W_CITY`: Skor evaluasi wilayah tempat tinggal.
* `WEEKDAY_APPR_PROCESS_START` & `HOUR_APPR_PROCESS_START`: Hari dan jam aplikasi diajukan.
* `ORGANIZATION_TYPE`: Jenis sektor industri tempat nasabah bekerja.

**3. Atribut Finansial Utama**
* `AMT_INCOME_TOTAL`: Total pendapatan tahunan nasabah.
* `AMT_CREDIT`: Total plafon pinjaman yang diajukan.
* `AMT_ANNUITY`: Biaya anuitas / besaran cicilan bulanan.
* `AMT_GOODS_PRICE`: Harga riil barang yang akan dibiayai pinjaman.

**4. Dokumen Legalitas (Flags)**
* Terdiri dari flag biner kontak: `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_EMAIL`.
* Terdiri dari 20 variabel dokumen verifikasi administrasi: `FLAG_DOCUMENT_2` hingga `FLAG_DOCUMENT_21`.

**5. Riwayat Kontak & Konsistensi Alamat**
* `DAYS_LAST_PHONE_CHANGE`: Jeda hari sejak pergantian nomor telepon terakhir.
* Rangkaian flag pencocokan alamat registrasi vs alamat aktual vs alamat kerja: `REG_REGION_NOT_LIVE_REGION`, `REG_REGION_NOT_WORK_REGION`, `LIVE_REGION_NOT_WORK_REGION`, `REG_CITY_NOT_LIVE_CITY`, `REG_CITY_NOT_WORK_CITY`, `LIVE_CITY_NOT_WORK_CITY`.

**6. Skor Evaluasi Pihak Ketiga (Biro Kredit Eksternal)**
* `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`: Skor probabilitas risiko yang dihitung oleh biro pemeringkat kredit eksternal. (Fitur ini merupakan penentu terkuat dalam *Exploratory Data Analysis*).

**7. Standar & Kondisi Bangunan Tempat Tinggal**
Dataset menyertakan puluhan variabel yang merinci spesifikasi fisik tempat tinggal nasabah (terbagi dalam format rata-rata `_AVG`, modus `_MODE`, dan median `_MEDI`):
* Metrik ukuran: `APARTMENTS_`, `BASEMENTAREA_`, `COMMONAREA_`, `LANDAREA_`, `LIVINGAPARTMENTS_`, `LIVINGAREA_`, `NONLIVINGAPARTMENTS_`, `NONLIVINGAREA_`, `TOTALAREA_MODE`.
* Metrik usia & struktur: `YEARS_BEGINEXPLUATATION_`, `YEARS_BUILD_`, `ELEVATORS_`, `ENTRANCES_`, `FLOORSMAX_`, `FLOORSMIN_`.
* Kondisi material & kelayakan: `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`.

**8. Kueri Rekam Jejak (Credit Bureau Enquiries)**
Frekuensi pengecekan nama nasabah oleh institusi keuangan lain sebelum pengajuan:
* `AMT_REQ_CREDIT_BUREAU_HOUR`, `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`, `AMT_REQ_CREDIT_BUREAU_MON`, `AMT_REQ_CREDIT_BUREAU_QRT`, `AMT_REQ_CREDIT_BUREAU_YEAR`.

**9. Pengamatan Dinamika Lingkungan Sosial**
* `OBS_30_CNT_SOCIAL_CIRCLE` & `DEF_30_CNT_SOCIAL_CIRCLE`: Keterlambatan dan gagal bayar di lingkaran sosial nasabah (30 hari).
* `OBS_60_CNT_SOCIAL_CIRCLE` & `DEF_60_CNT_SOCIAL_CIRCLE`: Keterlambatan dan gagal bayar di lingkaran sosial nasabah (60 hari).

### Visualisasi & Exploratory Data Analysis (EDA)
Dari hasil penelusuran data (EDA), ditemukan fakta kritis bahwa kelas `TARGET` terdistribusi pada persentase **~92% Kelas 0 (Lancar)** dan **~8% Kelas 1 (Macet)**. Selain itu, fitur `EXT_SOURCE` serta variabel finansial murni terbukti memiliki korelasi tertinggi dalam memisahkan batas kelas secara linear dan non-linear. Hal ini mengindikasikan perlunya rekayasa fitur lanjutan berbasis rasio finansial.

---

## Data Preparation

Teknik penyiapan data yang diaplikasikan berjalan secara berurutan pada *notebook* dengan rincian sebagai berikut:

1.  **One-Hot Encoding (OHE) untuk Fitur Kategorikal**
    * **Proses:** Mengonversi 12 kolom kategorikal menjadi kolom-kolom biner (0 dan 1).
    * **Alasan:** Algoritma *Machine Learning* yang digunakan membutuhkan masukan berupa format numerik murni agar komputasi matematika pemecahan simpul pohon (*tree node splitting*) dapat dijalankan.
2.  **Domain-Specific Feature Engineering**
    * **Proses:** Menciptakan fitur interaksi rasio finansial baru dari kolom yang sudah ada. Fitur yang dibuat meliputi `CREDIT_TO_INCOME_RATIO` (Total Kredit / Total Pendapatan), `ANNUITY_TO_INCOME_RATIO`, dan `GOODS_TO_CREDIT_RATIO`.
    * **Alasan:** Angka pendapatan atau plafon kredit secara absolut seringkali kurang representatif. Mengubahnya menjadi rasio akan merepresentasikan kemampuan bayar riil dan tingkat beban utang nasabah (contoh: rasio cicilan yang melebihi 30% dari gaji menunjukkan risiko tinggi).
3.  **Train-Test Split dengan Stratifikasi**
    * **Proses:** Membagi dataset utuh ke dalam *Training Set* (80%) dan *Test Set* (20%) menggunakan parameter wajib `stratify=y` dari `scikit-learn`.
    * **Alasan:** Untuk mencegah kebocoran data (*data leakage*) dan memastikan proporsi kelas minoritas (8% nasabah macet) terwakili dengan komposisi yang persis sama, baik di data latihan maupun di data ujian. Jika tidak dilakukan, model berisiko tidak mengenali kelas minoritas saat fase pengujian.

---

## Modeling

Proyek ini menggunakan komparasi dari tiga algoritma *Ensemble Tree-Based* untuk menyelesaikan masalah klasifikasi biner, dengan rincian:

### 1. LightGBM Classifier (Model Utama)
* **Kelebihan:** Sangat cepat dalam proses pelatihan (*training*) dan efisien memori. Menggunakan metode pertumbuhan daun (*leaf-wise growth*) yang sangat peka menangkap sinyal dari kelas minoritas pada dataset berskala besar.
* **Kekurangan:** Rentan terjadi *overfitting* jika parameter pembatas kedalaman daun tidak disetel secara akurat.
* **Proses Improvement (Hyperparameter Tuning):** Model dioptimasi secara matematis menggunakan *framework* pencarian **Optuna**. Rentang pencarian diarahkan pada kontrol kedalaman pohon dan regularisasi. Hasil parameter terbaik yang didapatkan dan digunakan pada tahap akhir adalah: `n_estimators=424`, `learning_rate=0.0168`, `num_leaves=57`, `max_depth=10`, dengan `class_weight='balanced'`. 
* **Threshold Customization:** Untuk meningkatkan keseimbangan akurasi bisnis, batas keputusan *predict_proba* dari *default* $0.5$ digeser ke batas optimal **$0.65$**.

### 2. XGBoost Classifier (Model Pembanding 1)
* **Tahapan & Parameter:** Dibangun dengan pendekatan pertumbuhan *level-wise* yang diagresifkan. Parameter disetel pada `n_estimators=300`, `max_depth=6`, dan kalkulasi `scale_pos_weight` otomatis (rasio 11.38). Threshold disamakan pada $0.65$.
* **Kelebihan:** Memiliki mekanisme regularisasi turunan L1/L2 yang sangat matang untuk memangkas *noise* pada data berdimensi banyak.
* **Kekurangan:** Lebih lambat secara komputasi dan rakus memori dibandingkan LightGBM.

### 3. Random Forest Classifier (Model Pembanding 2)
* **Tahapan & Parameter:** Membangun ratusan pohon terpisah (paralel) menggunakan konsep *Bagging*. Parameter disetel pada `n_estimators=150`, `max_depth=12`, dan `class_weight='balanced_subsample'`.
* **Kelebihan:** Paling stabil, *robust*, dan hampir tidak memerlukan penskalaan fitur tambahan.
* **Kekurangan:** Cenderung menghasilkan tebakan probabilitas yang mengelompok di nilai tengah, sehingga sulit mengidentifikasi batas tegas pada kasus ketimpangan kelas yang parah.

### Penentuan Model Terbaik
Berdasarkan hasil eksperimen, **LightGBM (Tuned)** dipilih sebagai **Model Terbaik**. LightGBM berhasil melampaui algoritma XGBoost dan Random Forest dengan mencapai skor ROC-AUC tertinggi dan menjaga *Recall* di tingkat yang jauh lebih aman secara struktural operasional.

---

## Evaluation

Dalam kasus perbankan dengan data imbalanced (di mana gagal bayar bertindak sebagai "Kelas 1 / Positif" dan lancar sebagai "Kelas 0 / Negatif"), metrik akurasi murni bisa sangat menyesatkan. Oleh karena itu, metrik yang dijadikan pedoman utama pada proyek ini adalah:

### Metrik Evaluasi yang Digunakan
1.  **ROC-AUC Score:** Metrik primer proyek. Mengukur keseluruhan probabilitas model membedakan Kelas 0 dan Kelas 1 dari nilai threshold $0.0$ hingga $1.0$.
2.  **Recall (Sensitivity):** Berapa banyak nasabah macet sejati yang berhasil dijaring oleh model dari keseluruhan nasabah yang aslinya gagal bayar.
    $$Recall = \frac{True\ Positive}{True\ Positive + False\ Negative}$$
3.  **Precision:** Dari seluruh nasabah yang ditebak gagal bayar oleh model, berapa persentase yang aslinya benar-benar macet.
    $$Precision = \frac{True\ Positive}{True\ Positive + False\ Positive}$$
4.  **Accuracy:** Kalkulasi klasifikasi kebenaran secara komprehensif.
    $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Hasil Proyek Berdasarkan Metrik Evaluasi

Evaluasi pada data *Test Set* menghasilkan tabel matriks berikut:

| Model | Threshold | ROC-AUC | Accuracy | Precision (Cl. 1) | Recall (Cl. 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **0.65** | **0.7628** | **84.41%** | **0.2416** | **43.54%** |
| XGBoost | 0.65 | 0.7581 | 85.04% | 0.2449 | 40.99% |
| Random Forest | 0.60 | 0.7333 | 86.91% | 0.2466 | 30.25% |

**Kesimpulan Evaluasi & Dampak Bisnis:**
* Model **LightGBM (Tuned)** berhasil mendominasi performa dengan **ROC-AUC Score tertinggi di 0.7628** dan kemampuan penangkapan (*Recall*) sebesar 43.54%.
* *Insight* paling fundamental dari proyek ini terbukti melalui modifikasi *Threshold Customization*. Dengan menggeser threshold LightGBM ke angka **$0.65$**, performa *Accuracy* global model berhasil terkatrol menjadi 84.41%.
* Secara operasional, penggeseran ini sukses **menyelamatkan sistem bank dari penolakan buta terhadap 10.345 calon nasabah yang lancar** (penurunan nilai *False Positive* secara tajam jika dibandingkan dengan model baseline bawaan).
* Melalui perpaduan algoritma berkinerja tinggi, injeksi fitur domain finansial, dan optimasi *hyperparameter*, proyek ini berhasil melahirkan sistem *Credit Scoring* yang matang dan memenuhi standardisasi penyelesaian masalah operasional perbankan di dunia riil.