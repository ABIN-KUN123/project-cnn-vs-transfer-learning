# Panduan Esensial & Rangkuman Proyek: CNN vs Transfer Learning

Dokumen ini berisi rangkuman alur berpikir, logika desain arsitektur, dan analisis hasil eksperimen riil sebagai panduan cepat (*read me*) untuk menguasai materi proyek sebelum dikumpulkan ke dosen.

---

## 🧭 1. Inti Masalah: Apa yang Sedang Diuji?

Proyek ini membandingkan dua paradigma besar dalam melatih jaringan saraf dalam (*deep learning*) untuk mengenali gambar:
* **CNN from Scratch (Belajar dari Nol):** Diuji menggunakan subset biner dari dataset **CIFAR-10** (Kelas *Airplane* dan *Automobile*). Pendekatan ini menguji performa model ketika dipaksa mempelajari filter abstraksi visual secara mandiri dari jumlah data yang sangat terbatas.
* **Transfer Learning (Memanfaatkan Pakar):** Diuji menggunakan dataset **Cats vs Dogs**. Pendekatan ini memanfaatkan arsitektur **MobileNetV2** yang telah memiliki filter visual matang hasil pelatihan menggunakan jutaan gambar pada dataset skala besar ImageNet. Proses latihan hanya berfokus pada penyesuaian lapisan penalaran terakhir (*top classifier*).

---

## 🏗️ 2. Alur Arsitektur Model (Desain Eksperimen)

### A. CNN from Scratch (CIFAR-10)
Model disusun secara sekuensial dari lapisan bawah ke atas dengan konfigurasi sebagai berikut:
1. **3x Layer Konvolusi (`Conv2D`):** Bertahap menggunakan 32, 64, dan 64 filter berukuran $3 \times 3$ dengan aktivasi `ReLU`. Layer awal mendeteksi pola dasar (garis/tepi objek), sedangkan layer akhir mendeteksi bentuk semantik yang lebih kompleks.
2. **2x Layer Pooling (`MaxPooling2D`):** Berfungsi mereduksi dimensi spasial gambar sebesar $2 \times 2$ agar komputasi lebih efisien.
3. **Flatten & Dense Layer:** Mengubah matriks fitur 2D menjadi vektor linear 1D untuk diproses oleh lapisan terhubung penuh (*fully connected*) berisi 64 neuron.
4. **Dropout (0.5):** Mematikan 50% neuron secara acak selama training untuk memaksa model melakukan generalisasi pola dan menekan laju hafalan mentah (*overfitting*).
5. **Output Layer:** Menggunakan 1 unit neuron dengan fungsi aktivasi `Sigmoid` untuk menghasilkan probabilitas klasifikasi biner.

### B. Transfer Learning (Cats vs Dogs)
1. **Base Model:** Memakai arsitektur MobileNetV2 dengan bobot pra-latih bawaan ImageNet.
2. **Strategi Feature Extraction:** Mengunci seluruh layer dasar (`base_model.trainable = False`) agar representasi fitur visual cerdas yang sudah ada tidak rusak/berubah.
3. **Top Classifier:** Menambahkan lapisan penalaran baru (*Global Average Pooling*, *Dense 64 ReLU*, *Dropout 0.5*, dan *Output Dense Sigmoid*) untuk dilatih khusus mengenali objek kucing dan anjing.

---

## 📊 3. Analisis Hasil Berdasarkan Data Riil Notebook

Berdasarkan berkas log `.ipynb` yang telah dieksekusi menggunakan GPU Tesla T4 di Kaggle, berikut kesimpulan performa angka eksak kedua model:

### A. Isu Overfitting pada CNN from Scratch
* **Akurasi Latih (*Training Accuracy*):** Berhasil menyentuh angka **84.07%** pada epoch ke-5.
* **Akurasi Uji (*Testing Accuracy*):** Tertahan di angka **79.32%**.
* **Analisis Singkat:** Terjadi jurang pemisah (*gap*) yang melebar di mana akurasi latih jauh mengungguli validasi/testing. Ini bukti empiris bahwa dataset biner minimal (200 data) **tidak cukup besar** untuk melatih CNN dari nol. Model cenderung mengalami *overfitting* (menghafal data latih dan gagal mengenali data baru).

### B. Superioritas Transfer Learning
* **Akurasi Uji (*Testing Accuracy*):** Melesat tinggi hingga **93.92%** hanya dalam waktu 3 epoch.
* **Waktu Training:** Jauh lebih singkat, yaitu **17.85 detik** dibanding CNN dari awal yang memakan waktu 40.12 detik.
* **Analisis Singkat:** Model tampil sangat efisien dan stabil. Karena pondasi filternya sudah matang mengenali bentuk-bentuk objek di dunia nyata, model mampu memberikan akurasi superior meskipun disuplai dengan jumlah sampel data yang minim.

---

## 🏥 4. Rangkuman Pengambilan Keputusan Studi Kasus

Argumentasi pemecahan masalah terhadap empat skenario riil yang diajukan oleh dosen:
1. **Skenario 1 (Klinik Medis, 300 Gambar):** Pilih **Transfer Learning**. Data medis sangat sensitif dan rentan terhadap kesalahan diagnosis fatal. Jumlah 300 data terlalu sedikit untuk melatih filter dari nol, sehingga meminjam bobot cerdas pretrained model jauh lebih aman.
2. **Skenario 2 (Perusahaan, 1 Juta Gambar Produk Unik):** **CNN from Scratch sangat relevan**. Pasokan 1 juta data sudah melimpah untuk melatih jaringan saraf dalam dari nol. Karakteristik produk internal yang jauh berbeda dari objek ImageNet menuntut pembuatan filter abstraksi visual baru secara mandiri.
3. **Skenario 3 (Prototipe Cepat, Tenggat Waktu 2 Hari):** Pilih **Transfer Learning (Feature Extraction)**. Pembekuan layer dasar memangkas waktu komputasi hingga hitungan detik, menjadikannya pilihan paling rasional untuk mengejar pembuktian kelayakan produk (*proof of concept*) secara instan.
4. **Skenario 4 (Infrastruktur Mewah, Data Besar + GPU Memadai):** **Transfer Learning tetap diperlukan** melalui teknik *Deep Fine-Tuning* (membuka sebagian layer atas untuk dilatih ulang bersama data baru). Memulai dari bobot pretrained bertindak sebagai inisialisasi cerdas (*smart initialization*) yang membuat konvergensi model tercapai jauh lebih cepat dibandingkan mengacak bobot dari nol.

---

## 📄 5. Refleksi Akhir

* **Tantangan Teknis:** Melakukan manipulasi pipeline data menggunakan `tf.data.Dataset` serta menyeimbangkan kembali pemrosesan nilai piksel akibat normalisasi yang berbeda di TensorFlow Datasets.
* **Kesimpulan Filosofis:** Performa kecerdasan buatan tidak semata-mata ditentukan oleh kompleksitas baris kode arsitektur program, melainkan ditentukan oleh keselarasan yang tepat antara volume ketersediaan data, batasan waktu, infrastruktur GPU, dan karakteristik objek nyata yang ingin dipecahkan.
