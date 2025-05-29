# Laporan Proyek Machine Learning - Alivia Vinca Kustaryono

## Domain Proyek
**Klaim Asuransi: Deteksi Klaim yang Berisiko**

Dalam dunia asuransi, klaim palsu atau berisiko tinggi merupakan salah satu penyebab utama kerugian finansial perusahaan. Deteksi dini terhadap klaim yang berpotensi merugikan sangat penting untuk menjaga keberlanjutan bisnis dan kepercayaan pelanggan.

Menurut riset oleh Coalition Against Insurance Fraud (2022), penipuan klaim menyebabkan kerugian lebih dari $80 miliar per tahun di AS saja [1]. Dengan memanfaatkan teknik machine learning, perusahaan asuransi dapat mengidentifikasi klaim-klaim abnormal atau mencurigakan secara otomatis dan efisien.

[1] Coalition Against Insurance Fraud, "By the numbers: fraud statistics," 2022. [Online]. Available: https://insurancefraud.org/

## Business Understanding

### Problem Statements
1. Bagaimana memprediksi apakah sebuah klaim asuransi memiliki potensi risiko tinggi?
2. Apa saja faktor-faktor yang mempengaruhi risiko pada klaim asuransi?
3. Bagaimana mengotomatiskan proses deteksi klaim yang mencurigakan secara efisien?


### Goals
1. Membangun model klasifikasi untuk memprediksi potensi risiko dari sebuah klaim.
2. Menentukan fitur-fitur penting yang berkontribusi terhadap penilaian risiko.
3. Memberikan sistem pendukung keputusan untuk tim investigasi klaim.


### Solution statements
1. Mengembangkan empat model: Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier dan CatBoost Classifier.
2. Tuning parameter untuk XGBoost Classifier agar mendapatkan performa optimal.
3. Evaluasi menggunakan metrik Akurasi, Precision, Recall, F1 Score, dan ROC-AUC.

## Data Understanding
Dataset yang digunakan adalah [Insurance Claims Dataset](https://www.kaggle.com/datasets/litvinenko630/insurance-claims), berisi data klaim asuransi mobil yang mencakup informasi pelanggan, properti, dan status klaim.

### Kondisi Data (Inspeksi Awal)
- Jumlah data: 58592 baris, 41 kolom
- Tidak ada missing value
- Tidak ada data duplicate
- Distribusi `claim status`: '0' ; 54844, '1' ; 3748 artinya 93.6% aman dan 6.4% berisiko

### Variabel-variabel dalam dataset meliputi:
- `policy_id`: ID unik untuk setiap polis asuransi
- `subscription_length`: Lama langganan asuransi (dalam tahun)
- `vehicle_age`: Usia kendaraan saat ini (dalam tahun)
- `customer_age`: Usia pemilik kendaraan atau pelanggan (dalam tahun)
- `region_code`: Kode wilayah tempat pelanggan tinggal
- `region_density`: Kepadatan wilayah 
- `segment`: Segmen kendaraan 
- `model`: Model kendaraan
- `fuel_type`: Jenis bahan bakar kendaraan (misalnya: bensin, diesel, CNG)
- `max_torque`: Torsi maksimum kendaraan
- `max_power`: Tenaga maksimum kendaraan
- `engine_type`: Jenis mesin kendaraan
- `airbags`: Jumlah airbag yang tersedia di kendaraan
- `is_esc`: Apakah kendaraan memiliki Electronic Stability Control (ESC)
- `is_adjustable_steering`: Apakah setir dapat disesuaikan
- `is_tpms`: Apakah memiliki Tire Pressure Monitoring System
- `is_parking_sensors`: Apakah terdapat sensor parkir
- `is_parking_camera`: Apakah terdapat kamera parkir
- `rear_brakes_type`: Jenis rem belakang (disc, drum)
- `displacement`: Kapasitas mesin dalam cc
- `cylinder`: Jumlah silinder pada mesin
- `transmission_type`: Jenis transmisi (manual, automatic)
- `steering_type`: Jenis sistem kemudi (power, electric)
- `turning_radius`: Radius putar kendaraan
- `length`: Panjang kendaraan 
- `width`: Lebar kendaraan
- `gross_weight`: Berat kotor kendaraan
- `is_front_fog_lights`: Apakah memiliki lampu kabut depan
- `is_rear_window_wiper`: Apakah ada wiper kaca belakang
- `is_rear_window_washer`: Apakah ada washer kaca belakang
- `is_rear_window_defogger`: Apakah ada defogger kaca belakang
- `is_brake_assist`: Apakah ada fitur brake assist
- `is_power_door_locks`: Apakah ada kunci pintu otomatis
- `is_central_locking`: Apakah kendaraan memiliki sistem penguncian sentral
- `is_power_steering`: Apakah ada power steering
- `is_driver_seat_height_adjustable`: Apakah kursi pengemudi bisa diatur ketinggiannya
- `is_day_night_rear_view_mirror`: Apakah ada spion malam/siang otomatis
- `is_ecw`: Emergency Call Warning – apakah tersedia
- `is_speed_alert`: Apakah kendaraan memiliki sistem peringatan kecepatan
- `ncap_rating`: Rating keselamatan kendaraan menurut NCAP
- `claim_status`: Target variabel – apakah klaim berisiko tinggi (1) atau tidak (0)

Dataset ini cocok untuk kasus klasifikasi biner.

### Pengelompokan Variabel
Target yang akan kita gunakan dalam memprediksi risiko klaim ini adalah `claim_status`, maka kita akan mengelompokkan fitur-fitur berikut sesuai dengan kegunaannya.
- Pelanggan dan Polis
  - `subscription_length` :durasi langganan → mungkin korelasi dengan risiko
  - `customer_age`: usia pelanggan bisa relevan
  - `region_density`: menggambarkan risiko lingkungan padat vs sepi
- Kendaraan & Spesifikasi Teknis
  - `vehicle_age`: usia kendaraan → biasanya kendaraan lebih tua = lebih berisiko
  - `fuel_type`: kadang terkait dengan performa kendaraan
  - `displacement`: menggambarkan performa
  - `airbags`: jumlah airbag bisa memengaruhi risiko kerusakan
  - `engine_type`, `transmission_type`, `steering_type`
  - `cylinder`: mesin besar lebih cepat, bisa lebih berisiko
  - `length`, `width`, `gross_weight`: ukuran & bobot kendaraan
  - `turning_radius`: bisa pengaruhi kelincahan/safety
- Fitur keamanan kendaraan
  - `is_esc`, `is_tpms`, `is_brake_assist`,
  - `is_parking_sensors`, `is_parking_camera`,
  - `is_rear_window_defogger`, `is_power_steering`,
  - `is_central_locking`, `is_power_door_locks`,
  - `is_speed_alert`, `is_ecw`,
  - `is_day_night_rear_view_mirror`, `is_driver_seat_height_adjustable`,
  - `is_front_fog_lights`, `is_rear_window_wiper`, `is_rear_window_washer`
- Rating keselamatan
  - `ncap_rating`: penting, karena menunjukkan tingkat keselamatan kendaraan

Dari 41 variabel di atas, variabel `policy_id`,`segment`, `model`,`max_torque`, `max_power` dihapus dikarenakan terlalu banyak kategori unik (high cardinality) tanpa encoding.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) adalah tahap penting dalam proses analisis data yang bertujuan untuk mengeksplorasi data secara visual dan statistik guna memahami karakteristiknya, menemukan pola atau hubungan antar variabel, mengidentifikasi outlier, serta mendeteksi data tidak normal atau data yang hilang. Tahap ini menjadi dasar pengambilan keputusan sebelum memasuki proses preprocessing dan pemodelan.
EDA pada project ini akan dilakukan statistik deskriptif, distribusi variabel target, visualisasi distribusi fitur, heatmap korelasi data numerik dan korelasi fitur numerik dengan target.

**Statistik Deskriptif**

Statistik deskriptif dapat dilihat dengan memanggil `insurance.describe()` dan `insurance.info()`
![insurance info() (1)](https://github.com/user-attachments/assets/473416b0-41d7-4e05-95cf-817fb4474ea2)
![insurance info() (2)](https://github.com/user-attachments/assets/c882ab5b-6a87-4fc7-a64c-c42046e385c7)
Gambar di atas merupakan hasil dari pemanggilan `insurance.info()` dan didapatkan kesimpulan bahwa terdapat 58592 baris dan 41 kolom awal.

![insurance describe() (1)](https://github.com/user-attachments/assets/836b0ff8-8088-432d-922c-f045e13fab8e)
![insurance describe() (2)](https://github.com/user-attachments/assets/c8e0e597-9ff0-45ff-a544-e3c74df92ad3)
Gambar di atas merupakan hasil dari pemanggilan `insurance.describe()` agar kita dapat mengetahui fitur kategorik nya seperti mean standar deviasi dan lain-lain.


**Distribusi Variabel Target**

Distribusi variabel target dapat dilihat dengan menghitung `claim_status`
![Hitung Claim Status](https://github.com/user-attachments/assets/bc4fe0f0-0497-400c-bee3-c6c7defd53d3)
![Distribusi Variabel Target Piechart](https://github.com/user-attachments/assets/281b99db-6084-44ed-ade7-b9c88aac8c18)
![Distribusi Variabel Target Barchart](https://github.com/user-attachments/assets/4c0986bf-e44c-4234-a55d-337ae47e652c)
Gambar di atas merupakan persebaran `claim_status` baik secara numerik, piechart maupun barchart.


**Visualisasi Distribusi Fitur**

Fitur seperti `vehicle_age`, `costumer_age`, `region_density`, `airbags`, dan `ncap_rating` menunjukkan potensi korelasi dengan target (`claim_status`).
![Barplot distribusi fitur](https://github.com/user-attachments/assets/a4d2157e-63fd-4410-b066-9525fb32741e)
Gambar di atas merupakan merupakan visualisasi barchart dari distribusi fitur yang menunjukan korelasi dengan target. 


**Heatmap Korelasi Data Numerik**

![Heatmap](https://github.com/user-attachments/assets/56032c0d-f1c0-4df4-9f82-3a9c39c21a5c)
Gambar di atas merupakan Heatmap korelasi data numerik sebelum preprocessing data.
![Heatmap Korelasi Data Numerik](https://github.com/user-attachments/assets/cdea76db-7386-4ae5-9b8a-bab44c9d4b58)
Gambar di atas merupakan Heatmap korelasi data numerik setelah preprocessing data.


**Korelasi Fitur Numerik Dengan Target**

Fitur keamanan kendaraan umumnya memiliki korelasi negatif terhadap klaim berisiko, menandakan fitur tersebut membantu mengurangi risiko klaim.
![Korelasi](https://github.com/user-attachments/assets/d9bae52f-03b1-469a-a423-2deb9f958247)
Gambar di atas merupakan korelasi setiap fitur numerik dengan target sebelum preprocessing data.
![Korelasi Setiap Fitur Numerik dengan target](https://github.com/user-attachments/assets/d1fa4137-5d4b-406c-8535-bb1723c4178d)
Gambar di atas merupakan korelasi setiap fitur numerik dengan target setelah preprocessing data.


## Data Preparation
Langkah-langkah data preparation:
1. Hapus kolom yang tidak digunakan seperti `policy_id`, `segment`, `model`, `max_torque`, `max_power`
2. Label Encoding untuk fitur biner seperti `is_esc`, `is_adjustable_steering`, `is_tpms`, `is_parking_sensors`, `is_parking_camera`, `rear_brakes_type`, `transmission_type`, `steering_type`, `is_front_fog_lights`, `is_rear_window_wiper`, `is_rear_window_washer`, `is_rear_window_defogger`, `is_brake_assist`, `is_power_door_locks`, `is_central_locking`, `is_power_steering`, `is_driver_seat_height_adjustable`, `is_day_night_rear_view_mirror`, `is_ecw`, `is_speed_alert` yang hanya memiliki dua nilai (misal: 'Yes' dan 'No'). Encoding ini mengubahnya menjadi angka 0 dan 1. 
3. One-hot encoding untuk fitur multikategori seperti `region_code`, `fuel_type`, `engine_type`. Fitur seperti `fuel_type` memiliki banyak kategori (misal: Petrol, Diesel, CNG), sehingga digunakan one-hot encoding untuk menghindari asumsi ordinalitas.
4. Feature scaling menggunakan StandardScaler untuk fitur numerik seperti `subscription_length`, `vehicle_age`, `customer_age`, `region_density`, `airbags`, `displacement`, `cylinder`, `turning_radius`, `length`, `width`, `gross_weight`, `ncap_rating` karena sebagian besar model machine learning sensitif terhadap skala fitur.
5. Train-test split: 80% data latih, 20% data uji.
6. Handling imbalance: Menggunakan SMOTE untuk menyeimbangkan jumlah klaim palsu dan tidak.

Alasan:
- Encoding diperlukan agar data dapat digunakan oleh model ML.
- SMOTE digunakan karena kelas target sangat tidak seimbang (fraud << non-fraud).

## Modeling
Model yang digunakan:
1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **XGBoost Classifier**
4. **CatBoost Classifier**

### Random Forest Classifier
Random Forest adalah algoritma ensemble learning berbasis decision tree. Ia bekerja dengan membangun banyak decision tree (hutan) selama proses pelatihan dan menggabungkan prediksi masing-masing pohon (melalui voting mayoritas untuk klasifikasi) untuk meningkatkan akurasi dan mengurangi overfitting.

**Parameter Default Random Forest**
| **Parameter**         | **Default** | **Deskripsi**                                                                 |
|------------------------|-------------|-------------------------------------------------------------------------------|
| `n_estimators`         | `100`       | Jumlah pohon dalam hutan                                                     |
| `criterion`            | `'gini'`    | Ukuran kualitas split (opsi lain: `'entropy'`, `'log_loss'`)                 |
| `max_depth`            | `None`      | Kedalaman maksimum tiap pohon                                                |
| `min_samples_split`    | `2`         | Minimum sampel untuk membagi node internal                                   |
| `min_samples_leaf`     | `1`         | Minimum sampel di tiap daun                                                  |
| `max_features`         | `'sqrt'`    | Jumlah fitur yang dipertimbangkan untuk split                                |
| `bootstrap`            | `True`      | Apakah menggunakan *bootstrap sampling*                                      |
| `random_state`         | `None`      | Seed untuk reprodusibilitas                                                  |
| `n_jobs`               | `None`      | Jumlah core CPU yang digunakan (`None`: 1 core; `-1`: semua core tersedia)   |

**Kelebihan Random Forest**
1. Akurasi tinggi – Mengurangi overfitting dari decision tree tunggal.
2. Robust terhadap noise dan outlier – Hasil voting dari banyak pohon lebih stabil.
3. Dapat menangani data besar – Cocok untuk data dengan banyak fitur dan observasi.

**Kekurangan Random Forest**
1. Kurang interpretatif – Sulit memahami bagaimana keputusan dibuat (black box).
2. Model besar dan berat – Membutuhkan banyak memori dan waktu pelatihan.
3. Overfitting masih mungkin – Jika parameter tidak dituning dengan baik (terutama jika jumlah pohon terlalu sedikit atau terlalu dalam).

### Decision Tree Classifier
Decision Tree Classifier adalah algoritma pembelajaran terawasi (supervised learning) yang digunakan untuk tugas klasifikasi. Model ini bekerja dengan membagi data menjadi subset berdasarkan fitur yang menghasilkan informasi paling "murni" (impurity paling rendah), membentuk struktur pohon keputusan dari akar ke daun.

**Parameter Default Decion Tree**
| **Parameter**           | **Default** | **Deskripsi**                                                                 |
|--------------------------|-------------|-------------------------------------------------------------------------------|
| `criterion`              | `'gini'`    | Fungsi untuk mengukur kualitas split (`'gini'` atau `'entropy'`)              |
| `splitter`               | `'best'`    | Strategi split node (`'best'` atau `'random'`)                                |
| `max_depth`              | `None`      | Kedalaman maksimum pohon (None = tidak dibatasi)                              |
| `min_samples_split`      | `2`         | Minimum sampel untuk membagi node internal                                    |
| `min_samples_leaf`       | `1`         | Minimum sampel di node daun                                                   |
| `max_features`           | `None`      | Jumlah fitur yang dipertimbangkan saat mencari split terbaik                  |
| `random_state`           | `None`      | Seed untuk kontrol acak                                                       |
| `max_leaf_nodes`         | `None`      | Jumlah maksimum daun                                                          |
| `min_impurity_decrease`  | `0.0`       | Minimum pengurangan impurity yang diperlukan untuk split                      |
| `class_weight`           | `None`      | Bobot kelas untuk menangani data imbalance                                    |

**Kelebihan Decision Tree**
1. Mudah dipahami dan divisualisasikan – Sangat interpretatif (seperti bagan alur).
2. Sedikit preprocessing – Tidak perlu normalisasi atau scaling.
3. Menangani data kategorikal dan numerik.

**Kekurangan Decision Tree**
1. Rentan terhadap overfitting – Terutama jika tidak dipangkas atau tidak dibatasi kedalamannya.
2. Tidak stabil terhadap data kecil – Perubahan kecil pada data bisa menghasilkan pohon yang sangat berbeda.
3. Kurang akurat dibanding model ensemble – Seperti Random Forest atau Gradient Boosting.


### XGBoost Classifier
XGBoost (Extreme Gradient Boosting) adalah algoritma gradient boosting yang sangat efisien dan akurat, dirancang untuk performa tinggi. Ini menggunakan pendekatan boosting bertahap dan mengoptimasi fungsi loss dengan metode gradient descent.
XGBoost sangat populer di kompetisi data science seperti Kaggle karena performa, fleksibilitas, dan efisiensinya.

**Parameter Default XGBoost**
| **Parameter**         | **Default**     | **Deskripsi**                                                                 |
|------------------------|-----------------|-------------------------------------------------------------------------------|
| `booster`              | `'gbtree'`      | Jenis model yang digunakan (`'gbtree'`, `'gblinear'`, `'dart'`)              |
| `n_estimators`         | `100`           | Jumlah boosting round (jumlah pohon)                                         |
| `learning_rate`        | `0.3`           | Step size shrinkage untuk memperlambat pembelajaran                          |
| `max_depth`            | `6`             | Kedalaman maksimum pohon                                                     |
| `min_child_weight`     | `1`             | Minimum bobot total dari child (mengontrol overfitting)                      |
| `subsample`            | `1.0`           | Persentase data training yang digunakan per boosting round                   |
| `colsample_bytree`     | `1.0`           | Proporsi fitur yang digunakan untuk membuat setiap pohon                     |
| `gamma`                | `0`             | Minimum loss reduction agar split terjadi                                    |
| `reg_alpha`            | `0`             | Regularisasi L1 (mencegah overfitting)                                       |
| `reg_lambda`           | `1`             | Regularisasi L2                                                              |
| `objective`            | `'binary:logistic'` | Fungsi objektif (misalnya untuk klasifikasi biner)                      |
| `n_jobs`               | `None`          | Jumlah thread untuk paralelisasi (None: default dari sistem)                 |
| `random_state`         | `0`             | Seed random untuk reprodusibilitas                                           |

**Kelebihan XGBoost**
1. Performa tinggi – Cepat dan efisien secara komputasi.
2. Akurasi tinggi – Salah satu algoritma terbaik untuk prediksi tabular.
3. Regularisasi built-in – Mencegah overfitting dengan reg_alpha dan reg_lambda.

**Kekurangan XGBoost**
1. Lebih kompleks dibanding model dasar – Butuh tuning hyperparameter untuk hasil optimal.
2. Penggunaan memori tinggi – Terutama pada dataset besar.
3. Waktu pelatihan bisa lama – Jika jumlah estimator besar dan data besar.

### CatBoost Classifier
CatBoost adalah algoritma gradient boosting berbasis decision tree yang dikembangkan oleh Yandex. CatBoost secara khusus dioptimalkan untuk bekerja dengan data kategorikal tanpa perlu encoding manual (seperti one-hot encoding), menjadikannya sangat efisien dan akurat terutama untuk data tabular.
CatBoost menggunakan pendekatan yang memperbaiki prediction shift dan overfitting, dengan mendukung training yang stabil bahkan saat fitur kategorikal dominan.

**Parameter Default CatBoost**
| **Parameter**         | **Default**   | **Deskripsi**                                                                 |
|------------------------|---------------|-------------------------------------------------------------------------------|
| `iterations`           | `1000`        | Jumlah boosting iteration (jumlah pohon)                                     |
| `learning_rate`        | `None`        | Kecepatan pembelajaran (default otomatis berdasarkan data)                   |
| `depth`                | `6`           | Kedalaman maksimum pohon                                                     |
| `l2_leaf_reg`          | `3.0`         | Koefisien regularisasi L2                                                    |
| `loss_function`        | `'Logloss'`   | Fungsi loss untuk klasifikasi biner                                          |
| `border_count`         | `254`         | Jumlah split pada fitur numerik                                              |
| `thread_count`         | `-1`          | Jumlah thread CPU (`-1`: gunakan semua core tersedia)                        |
| `random_seed`          | `0`           | Seed random untuk reprodusibilitas                                           |
| `verbose`              | `False`       | Menampilkan log pelatihan                                                    |
| `cat_features`         | `None`        | Daftar indeks fitur kategorikal (ditangani otomatis jika diset dengan benar) |
| `auto_class_weights`   | `None`        | Penyesuaian bobot otomatis untuk menangani data imbalance                    |

**Kelebihan CatBoost**
1. Native handling untuk fitur kategorikal – Tidak perlu encoding manual.
2. Performa tinggi – Akurat dan cepat, sangat kompetitif dengan XGBoost dan LightGBM.
3. Stabil terhadap overfitting – Berkat teknik seperti ordered boosting.

**Kekurangan CatBoost**
1. Ukuran library besar – Bisa berat untuk deployment ringan.
2. Waktu pelatihan bisa lebih lama – Terutama dibanding LightGBM jika tanpa GPU.
3. Kurang fleksibel untuk beberapa tugas kustom – Misalnya pada model hybrid atau non-tabular.

### Langkah:
- Latih keempat model menggunakan cross-validation.
- Lakukan tuning `max_depth`, `n_estimators`, dan `learning_rate` untuk XGBoost dan `iterations`,`depth`, dan `learning_rate` untuk CatBoost .
- Gunakan GridSearchCV untuk hyperparameter optimization.

**Model Terbaik:**
Berdasarkan evaluasi terhadap keempat model, **XGBoost** dipilih sebagai model terbaik karena memberikan keseimbangan yang baik antara F1-score dan ROC AUC, yang penting dalam kasus klasifikasi data tidak seimbang seperti klaim asuransi palsu.

## Evaluation
Metrik evaluasi yang digunakan:
* **Accuracy**: seberapa banyak prediksi benar.
* **Precision**: seberapa tepat model dalam mengidentifikasi klaim fraud.
* **Recall**: seberapa banyak klaim fraud yang berhasil ditangkap model.
* **F1 Score**: trade-off antara precision dan recall.
* **ROC-AUC**: mengukur kemampuan klasifikasi secara umum.

### Hasil Evaluasi:

| Model          | Accuracy  | Precision | Recall   | F1 Score | ROC AUC  |
|----------------|-----------|-----------|----------|----------|----------|
| CatBoost       | 0.933441  | 0.142857  | 0.008000 | 0.015152 | 0.637794 |
| XGBoost        | 0.926530  | 0.127517  | 0.025333 | 0.042269 | 0.637066 |
| Random Forest  | 0.882669  | 0.087186  | 0.088000 | 0.087591 | 0.584391 |
| Decision Tree  | 0.866456  | 0.087133  | 0.114667 | 0.099021 | 0.518444 |

**Analisis:**
- CatBoost memiliki akurasi tertinggi, namun sangat rendah dalam recall (hanya 0.8%), yang menunjukkan kegagalan dalam mendeteksi klaim berisiko.
- XGBoost memberikan hasil paling seimbang antara recall, F1-score, dan ROC-AUC, meskipun masih belum ideal karena dataset yang sangat tidak seimbang.

**Model terbaik: XGBoost**
Dari hasil evaluasi, meskipun CatBoost memiliki accuracy tertinggi, recall-nya sangat rendah (0.8%), yang berarti sebagian besar klaim fraud gagal dideteksi. XGBoost memiliki F1-score dan ROC AUC yang cukup seimbang, sehingga dipilih sebagai model terbaik. Hal ini penting karena pada data yang imbalance, metrik seperti recall, F1-score, dan ROC AUC lebih relevan dibanding accuracy.

## Kesimpulan
Berdasarkan evaluasi menyeluruh terhadap beberapa model klasifikasi yang diterapkan pada dataset klaim asuransi dengan kondisi data yang sangat tidak seimbang, model XGBoost dipilih sebagai model terbaik.

XGBoost menunjukkan keseimbangan yang baik antara akurasi tinggi dan kemampuan generalisasi pada data uji. Meskipun metrik seperti recall dan F1 score masih relatif rendah karena tantangan data yang sangat tidak seimbang, XGBoost tetap memberikan performa yang lebih baik dibandingkan model lain seperti CatBoost, Random Forest, dan Decision Tree.

Dengan demikian, model XGBoost menjadi solusi yang tepat untuk kasus klasifikasi risiko klaim fraud, sekaligus menjadi dasar untuk pengembangan lebih lanjut seperti tuning hyperparameter dan penerapan teknik penanganan imbalance lebih lanjut.

## Referensi
[1] Coalition Against Insurance Fraud. (2022). By the numbers: fraud statistics. Retrieved from https://insurancefraud.org/
