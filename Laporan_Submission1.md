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
* **Jawaban Pernyataan Masalah 2:** Melakukan optimasi pada nilai *threshold* probabilitas keputusan model dan mengevaluasinya secara komprehensif menggunakan matriks *Confusion Matrix* dan evaluasi metrik turunan, sehingga performa model selaras dengan kebijakan selera risiko (*risk appetite*) bisnis.

### Solution statements
Untuk mencapai tujuan di atas, diusulkan pendekatan menggunakan tiga algoritma yang berbeda sebagai pembanding, dengan satu model utama yang akan dioptimasi:
1.  **Baseline Model 1 (Random Forest):** Menggunakan model Random Forest dengan parameter *bagging* `class_weight='balanced_subsample'` untuk menangani rasio kelas minoritas secara paralel.
2.  **Baseline Model 2 (XGBoost):** Menggunakan *Extreme Gradient Boosting* dengan kalkulasi parameter penyeimbang kelas `scale_pos_weight`.
3.  **Model Utama (LightGBM):** Menggunakan LightGBM dengan penyetelan hyperparameter yang dikonfigurasi secara spesifik untuk menangani kelas minoritas (`class_weight='balanced'`). Model ini dipasangkan dengan *Custom Threshold Optimization* (disetel pada angka 0.65) untuk mencapai metrik evaluasi bisnis yang paling optimal.

Solusi di atas dievaluasi secara terukur menggunakan metrik spesifik yang relevan dengan kasus *imbalanced data*, yakni **ROC-AUC, Recall, dan F1-Score**.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari kompetisi global **Home Credit Default Risk** yang dipublikasikan di platform Kaggle. Dataset utama (`application_train.csv`) memiliki dimensi yang sangat besar, terdiri dari 307.511 baris observasi dan 122 fitur awal sebelum tahap rekayasa data.

**Tautan Sumber Data:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

### Kondisi Data (Missing Value, Duplikat, dan Anomali)
Sebelum melakukan pemrosesan, dilakukan pemeriksaan kondisi data dengan temuan sebagai berikut:
* **Duplikat:** Tidak ditemukan adanya baris data yang terduplikasi. Seluruh `SK_ID_CURR` bersifat unik.
* **Missing Value (Nilai Kosong):** Dataset memiliki persentase *missing value* yang sangat tinggi pada beberapa kelompok kolom. Kolom standar bangunan tempat tinggal (seperti `APARTMENTS_AVG`, `BASEMENTAREA_AVG`, dll.) memiliki nilai kosong lebih dari 50%. Kolom skor eksternal (`EXT_SOURCE_1` dan `EXT_SOURCE_3`) juga memiliki nilai kosong sekitar 19% hingga 56%.
* **Anomali/Outlier:** Ditemukan anomali ekstrem pada kolom `DAYS_EMPLOYED` (durasi hari kerja). Terdapat sekitar 55.000+ baris yang bernilai positif `365243` (setara dengan 1.000 tahun), yang secara logika bisnis tidak mungkin dan merupakan penanda (*flag*) bawaan sistem lama yang mendandakan pengangguran atau pensiunan.

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

### Exploratory Data Analysis (EDA)
Dari hasil penelusuran data, distribusi target sangat timpang, yaitu **~92% Kelas 0** dan **~8% Kelas 1**. Fitur `EXT_SOURCE` serta variabel finansial murni memiliki korelasi tertinggi dalam memisahkan batas kelas. 

---

## Data Preparation

Tahapan penyiapan data dilakukan secara berurutan pada *notebook* untuk membersihkan, menangani anomali, serta mengubah struktur data agar siap digunakan oleh model *Machine Learning*:

1.  **Penanganan Nilai Anomali & Pembuatan Kolom Penanda (Flag)**
    * **Proses:** Mengganti nilai anomali `365243` pada kolom `DAYS_EMPLOYED` menjadi `NaN` (kosong), lalu membuat satu kolom *boolean* baru bernama `DAYS_EMPLOYED_ANOM` untuk menandai baris mana saja yang sebelumnya memiliki anomali tersebut.
    * **Alasan:** Angka 365243 akan merusak distribusi numerik model jika dibiarkan. Pembuatan kolom penanda menjaga agar "pola" atau informasi bahwa nasabah tersebut anomali (misalnya pensiunan) tidak hilang.
2.  **Imputasi Nilai Kosong (Missing Values)**
    * **Proses:** Melakukan pengisian data kosong. Untuk kolom numerik (seperti `AMT_ANNUITY` atau `EXT_SOURCE`), nilai kosong diisi dengan menggunakan nilai **Median**. Untuk kolom kategorikal, nilai kosong diisi menggunakan nilai **Modus** (kategori paling sering muncul).
    * **Alasan:** Model algoritma (seperti standar Random Forest `scikit-learn`) tidak bisa menerima *input* data kosong (`NaN`). Penggunaan median lebih tahan (*robust*) terhadap data *outlier* dibandingkan *mean*.
3.  **One-Hot Encoding (OHE)**
    * **Proses:** Mengonversi kolom-kolom kategorikal (seperti `CODE_GENDER` atau `NAME_INCOME_TYPE`) menjadi beberapa kolom biner dengan nilai 0 dan 1.
    * **Alasan:** Model matematika hanya bisa memproses kalkulasi berbasis angka numerik, sehingga label teks harus direpresentasikan secara matematis.
4.  **Domain-Specific Feature Engineering**
    * **Proses:** Menciptakan fitur interaksi rasio finansial baru, seperti `CREDIT_TO_INCOME_RATIO` (Total Kredit / Total Pendapatan) dan `ANNUITY_TO_INCOME_RATIO`.
    * **Alasan:** Mengubah nilai absolut menjadi rasio jauh lebih merepresentasikan daya beli dan beban utang riil calon nasabah.
5.  **Train-Test Split dengan Stratifikasi**
    * **Proses:** Membagi dataset menjadi *Training Set* (80%) dan *Test Set* (20%) menggunakan parameter `stratify=y`.
    * **Alasan:** Mengingat target kita hanya memiliki 8% nasabah macet, *stratify* sangat penting agar proporsi 8% tersebut terdistribusi persis sama rata di data latih dan data uji untuk mencegah bias pembelajaran.

---

## Modeling

Proyek ini menggunakan tiga algoritma *Ensemble Tree-Based* dengan pendefinisian parameter lengkap sebagai berikut:

### 1. LightGBM Classifier (Model Utama)
* **Kelebihan:** Sangat cepat dalam proses pelatihan (*training*) dan efisien memori berkat metode *leaf-wise growth*. Mampu menangani ribuan baris dengan cepat.
* **Kekurangan:** Rentan terjadi *overfitting* pada *noise* jika kedalaman dan daun pohon tidak dibatasi.
* **Parameter yang Digunakan:** * `n_estimators=424`: Jumlah maksimum pohon keputusan yang dibangun.
  * `learning_rate=0.0168`: Kecepatan belajar model tiap iterasi (dibuat kecil agar lebih presisi).
  * `max_depth=10`: Kedalaman maksimal setiap pohon.
  * `num_leaves=57`: Jumlah batas maksimum daun dalam satu pohon.
  * `class_weight='balanced'`: Otomatis memperbesar bobot penalti saat model salah menebak nasabah macet (kelas minoritas).
  * `random_state=42`: Menjaga hasil tetap konsisten (*reproducible*).
* **Threshold Customization:** Batas keputusan probabilitas *predict_proba* digeser secara manual dari 0.5 menjadi **0.65** untuk mengurangi kerugian *False Positive*.

### 2. XGBoost Classifier (Model Pembanding 1)
* **Kelebihan:** Sangat presisi, perlindungan terhadap *overfitting* yang luar biasa karena memiliki regularisasi internal L1/L2.
* **Kekurangan:** Lebih berat dan lambat secara komputasi (menggunakan memori yang tinggi) dibandingkan LightGBM.
* **Parameter yang Digunakan:** * `n_estimators=300`: Jumlah maksimum pohon.
  * `learning_rate=0.02`: Kecepatan model dalam belajar.
  * `max_depth=6`: Batas kedalaman pohon.
  * `scale_pos_weight=11.38`: Dihitung manual dari rasio jumlah kelas negatif dibagi kelas positif untuk *imbalanced data*.
  * `subsample=0.8`: Rasio sampel data yang digunakan per pohon.
  * `colsample_bytree=0.8`: Rasio fitur yang disampling per pohon.
  * `eval_metric='logloss'`: Metrik evaluasi bawaan saat iterasi.
  * `n_jobs=-1`: Memaksimalkan seluruh inti prosesor komputer (*multi-threading*).
  * `random_state=42`: Untuk konsistensi hasil.

### 3. Random Forest Classifier (Model Pembanding 2)
* **Kelebihan:** Model berbasis *bagging* yang sangat stabil dan hampir tidak sensitif terhadap *outliers* numerik.
* **Kekurangan:** Kurang agresif dalam mendeteksi batas (margin) untuk data yang sangat timpang.
* **Parameter yang Digunakan:**
  * `n_estimators=150`: Jumlah pohon (hutan) yang diparalelkan.
  * `max_depth=12`: Pembatasan kedalaman maksimal per pohon.
  * `class_weight='balanced_subsample'`: Bobot dihitung ulang di setiap proses pengambilan sampel bootstrap secara terpisah per iterasi.
  * `n_jobs=-1`: Pemrosesan terparalelisasi.
  * `random_state=42`.

### Penentuan Model Terbaik
Berdasarkan eksperimen dan kalkulasi di metrik evaluasi final, **LightGBM** dengan penyesuaian threshold (0.65) dan parameter `class_weight='balanced'` dipilih sebagai model solusi terbaik karena menghasilkan skor sensitivitas dan klasifikasi (*ROC-AUC*) yang paling optimal di antara model lain.

---

## Evaluation

Dalam kasus perbankan (*imbalanced data*), evaluasi didasarkan tidak hanya pada akurasi semata. Metrik berikut digunakan:

1.  **ROC-AUC Score:** Mengukur probabilitas model dalam membedakan kelas 0 dan 1 di berbagai batas nilai. Nilai semakin mendekati 1 semakin baik.
2.  **Recall (Sensitivity):** Berapa persentase nasabah gagal bayar yang berhasil "tertangkap" model dari keseluruhan nasabah yang aslinya gagal bayar.
    $$Recall = \frac{True\ Positive}{True\ Positive + False\ Negative}$$
3.  **Precision:** Dari semua nasabah yang *dituduh* model gagal bayar, berapa persen yang aslinya memang gagal bayar.
    $$Precision = \frac{True\ Positive}{True\ Positive + False\ Positive}$$
4.  **Accuracy:** Persentase tebakan klasifikasi benar secara keseluruhan.
    $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
5.  **F1-Score:** Ini adalah metrik terpenting (selain ROC-AUC) untuk imbalanced data, karena F1-Score merupakan nilai *harmonic mean* (rata-rata harmonis) yang menyeimbangkan *Precision* dan *Recall*. Jika salah satu metrik (presisi atau recall) hancur, F1-Score akan langsung turun drastis.
    $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### Hasil Proyek Berdasarkan Metrik Evaluasi

Tabel rekapitulasi performa model (*Test Set*):

| Model | Threshold | ROC-AUC | Accuracy | Precision (Cl. 1) | Recall (Cl. 1) | F1-Score (Cl. 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.65** | **0.7628** | **84.41%** | **0.2416** | **43.54%** | **0.3108** |
| XGBoost | 0.65 | 0.7581 | 85.04% | 0.2449 | 40.99% | 0.3066 |
| Random Forest | 0.60 | 0.7333 | 86.91% | 0.2466 | 30.25% | 0.2717 |

**Kesimpulan:**
Model **LightGBM** memberikan perpaduan yang paling ideal. Meskipun akurasinya sedikit lebih rendah dibanding Random Forest (karena threshold diturunkan), model ini memiliki **ROC-AUC tertinggi (0.7628)** dan **F1-Score tertinggi (0.3108)** untuk kelas minoritas. Model berhasil mendeteksi potensi gagal bayar (*Recall*) sebanyak 43.54%, yang berarti secara langsung akan membantu menekan risiko kerugian penyaluran dana hingga mendekati setengah dari rata-rata default bawaan data.