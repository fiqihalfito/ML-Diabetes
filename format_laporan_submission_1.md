# Laporan Proyek Machine Learning - Fiqih Alfito

## Domain Proyek

Diabetes adalah penyakit kronis yang ditandai dengan kadar gula darah yang tinggi. Ini adalah masalah kesehatan global yang signifikan, dengan jutaan orang yang terkena diabetes di seluruh dunia. Penanganan diabetes memerlukan pengawasan ketat terhadap kadar glukosa darah serta pengelolaan gaya hidup yang sehat. Dalam upaya untuk mengatasi dan mengelola diabetes, teknologi *machine learning* telah menjadi alat yang sangat berguna dalam menganalisis dan memprediksi risiko diabetes serta membantu dalam pengambilan keputusan klinis.

Pengembangan model *machine learning* yang dapat memprediksi diabetes berdasarkan data klinis individu telah menjadi fokus penelitian yang signifikan. Tujuan utama dari pengembangan model ini adalah untuk membangun sistem yang dapat mengklasifikasikan dengan akurat apakah seseorang mengalami diabetes atau tidak.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan

Penyakit diabetes dapat mengakibatkan komplikasi penyakit, yang tentunya sangat berbahaya terhadap penderita diabetes. Oleh karena itu, sangat diperlukanya suatu teknologi yang dapat mendeteksi penyakit diabetes dengan tingkat analisis yang akurat, sehingga penyakit diabetes dapat ditangani lebih awal untuk mengurangi jumlah penderita, kecacatan, dan kematian. Karena efek diabetes dapat menyebabkan komplikasi bahkan kematian, dibutukan model prediksi untuk mengklasifikasikan seseorang mengidap penyakit diabetes untuk mengetahui seseorang mengidap penyakit diabetes atau tidak secara cepat.[(A. Supandi, 2022)](https://ejournal.indobarunasional.ac.id/index.php/jursima/article/view/396)

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, penulis akan mengembangkan sebuah sistem prediksi diabetes untuk menjawab permasalahan berikut.

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh untuk mengetahui seseorang yang mengalami diabetes?
- Apakah seseorang sedang mengalami diabetes atau tidak dengan karakteristik atau fitur tertentu? 

### Goals

Untuk  menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur yang paling berkorelasi atau berpengaruh terhadap status diabetes seseorang.
- Membuat model *machine learning* yang dapat memprediksi seseorang mengalami diabetes atau tidak dengan seakurat mungkin berdasarkan fitur-fitur yang ada.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements

- Untuk mengetahui fitur mana yang paling berkorelasi dalam mengetahui seseorang yang sedang mengalami diabetes yaitu dengan metode *Bivariate analysis* yang mana digunakan untuk membandingkan fitur diabetes dengan fitur lainnya.
- untuk mendapat hasil yang akurat, penelitian perlu menggunakan beberapa algoritma. Berhubung data yang digunakan bersifat klasifikasi, setiap model akan diukur dengan metrik ``accuracy_score()``. Skala ``accuracy_score()`` berada di rentang 1 sampai 0 yang mana 1 menunjukkan sangat akurat dan 0 sangat tidak akurat. Model yang menunjukan akurasi yang paling tinggi adalah model yang akurat dalam memprediksi diabetes.

## Data Understanding

Data yang digunakan dalam proyek ini adalah [**Dataset Prediksi Diabetes**](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Dataset prediksi diabetes adalah kumpulan data medis dan demografis dari 100.000 pasien, beserta status diabetesnya (positif atau negatif). Data tersebut mencakup fitur seperti *age*, *gender*, *body mass index (BMI)*, *hypertension*, *heart disease*, *smoking history*, *HbA1c level*, dan *blood glucose level*. Kumpulan data ini dapat digunakan untuk membangun model *machine learning* untuk memprediksi diabetes pada pasien berdasarkan riwayat medis dan informasi demografis mereka. 

### Variabel-variabel pada Dataset Prediksi Diabetes adalah sebagai berikut:

- Age: Usia merupakan faktor penting dalam memprediksi risiko diabetes. Seiring bertambahnya usia individu, risiko mereka terkena diabetes meningkat. Hal ini sebagian disebabkan oleh faktor-faktor seperti berkurangnya aktivitas fisik, perubahan kadar hormon, dan kemungkinan lebih tinggi untuk mengembangkan kondisi kesehatan lain yang dapat menyebabkan diabetes.

- Gender: Jenis kelamin dapat berperan dalam risiko diabetes, meskipun efeknya dapat bervariasi. Misalnya, wanita dengan riwayat diabetes gestasional (diabetes saat hamil) memiliki risiko lebih tinggi terkena diabetes tipe 2 di kemudian hari. Selain itu, beberapa penelitian menunjukkan bahwa pria mungkin memiliki risiko diabetes yang sedikit lebih tinggi dibandingkan wanita.

- Body Mass Index (BMI): BMI adalah ukuran lemak tubuh berdasarkan tinggi dan berat badan seseorang. Ini umumnya digunakan sebagai indikator status berat badan secara keseluruhan dan dapat membantu dalam memprediksi risiko diabetes. BMI yang lebih tinggi dikaitkan dengan kemungkinan lebih besar terkena diabetes tipe 2. Kelebihan lemak tubuh, terutama di sekitar pinggang, dapat menyebabkan resistensi insulin dan merusak kemampuan tubuh untuk mengatur kadar gula darah.

- Hypertension: Hipertensi, atau tekanan darah tinggi, adalah kondisi yang sering terjadi bersamaan dengan diabetes. Kedua kondisi tersebut memiliki faktor risiko yang sama dan dapat berkontribusi pada perkembangan satu sama lain. Memiliki hipertensi meningkatkan risiko terkena diabetes tipe 2 dan sebaliknya. Kedua kondisi tersebut dapat berdampak buruk pada kesehatan jantung.

- Heart Disease: Penyakit jantung, termasuk kondisi seperti penyakit arteri koroner dan gagal jantung, dikaitkan dengan peningkatan risiko diabetes. Hubungan antara penyakit jantung dan diabetes bersifat dua arah, artinya memiliki satu kondisi meningkatkan risiko berkembangnya kondisi lainnya. Ini karena mereka memiliki banyak faktor risiko yang sama, seperti obesitas, tekanan darah tinggi, dan kolesterol tinggi.

- Smoking History: Merokok merupakan faktor risiko diabetes yang dapat dimodifikasi. Merokok sigaret diketahui meningkatkan risiko diabetes tipe 2. Merokok dapat menyebabkan resistensi insulin dan merusak metabolisme glukosa. Berhenti merokok dapat secara signifikan mengurangi risiko terkena diabetes dan komplikasinya.

- HbA1c Level: HbA1c (hemoglobin terglikasi) adalah ukuran rata-rata kadar glukosa darah selama 2-3 bulan terakhir. Ini memberikan informasi tentang kontrol gula darah jangka panjang. Tingkat HbA1c yang lebih tinggi menunjukkan kontrol glikemik yang lebih buruk dan berhubungan dengan peningkatan risiko diabetes dan komplikasinya.

- Blood Glucose Level: Tingkat glukosa darah mengacu pada jumlah glukosa (gula) yang ada dalam darah pada waktu tertentu. Peningkatan kadar glukosa darah, terutama dalam keadaan puasa atau setelah mengonsumsi karbohidrat, dapat mengindikasikan gangguan regulasi glukosa dan meningkatkan risiko diabetes. Pemantauan rutin kadar glukosa darah penting dalam diagnosis dan pengelolaan diabetes

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

Terdapat beberapa tahapan dalam memahami dataset tersebut, yaitu:

1. Periksa duplikasi

    Kehadiran data duplikasi dalam set pelatihan dapat menghasilkan evaluasi yang tidak akurat, karena model sudah terbiasa dengan data yang sama berulang-ulang.

    library Pandas menyediakan fungsi untuk menghapus baris duplikat yaitu : 
    > `pd.drop_duplicates()`

2. periksa *missing value*

    Jika missing value tidak dikelola dengan benar, ini dapat menyebabkan kesalahan dalam analisis data, termasuk perhitungan statistik dan visualisasi. Hasil yang tidak akurat atau bias dapat mengarah pada kesimpulan yang salah atau interpretasi yang tidak benar.

    Untuk mengetahui apakah suatu fitur memiliki nilai kosong atau tidak, berikut fungsi yang digunakan :
    > `pd.isnull().sum()`

3. *Univariate Analysis*

    *Univariate analysis*, atau analisis univariat, adalah metode analisis statistik yang digunakan untuk memahami dan menganalisis satu fitur pada satu waktu. Tujuan utama dari analisis univariat adalah untuk mendapatkan wawasan dan pemahaman yang mendalam tentang suatu fitur secara terpisah. Pada kasus ini digunakan diagram batang dan histogram untuk menganalisis univariat.

    analisis univariate dibagi menjadi dua yaitu analisis fitur kategorial dan fitur numerik. fitur kategorial disini terdiri dari fitur ``'gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes'``. Untuk fitur numerik terdiri dari fitur ``'age', 'bmi', 'HbA1c_level', 'blood_glucose_level'``.

    Dari hasil analisis, maka diperoleh kesimpulan sebagai berikut:

    1. fitur `gender`. Dari seluruh pasien, jumlah perempuan lebih banyak dibandingan dengan laki-laki.
    2. fitur `hypertension`. Dari seluruh pasien, pasien dengan kondisi hipertensi hanya sedikit, sebesar 7.8% saja.
    3. fitur `heart_disease`. Dari seluruh pasien, pasien yang mempunyai penyakit jantung sangat sedikit yaitu sebesar 4.1%.
    4. fitur `smoking_history`. Dari seluruh pasien, 70% pasien tidak pernah merokok.
    5. fitur `diabetes`. Dari seluruh pasien, pasien yang mengalami diabetes sebesar 8.8%.
     
    
4. *Bivariate Analysis*

    *Bivariate analysis*, atau analisis bivariat, adalah metode analisis statistik yang digunakan untuk memahami hubungan atau interaksi antara dua fitur secara simultan. Dalam analisis bivariat, fokus diberikan pada bagaimana fitur satu mempengaruhi fitur lainnya.

    Dari hasil analisis, maka diperoleh kesimpulan sebagai berikut:

    - *Gender* dan *Diabetes*

        Jumlah perempuan dan laki-laki yang mengalami diabetes hampir setara, perempuan sedikit lebih banyak dari laki-laki.
    
    - *Age* dan *Diabetes*

        Kecenderungan untuk terkena diabetes perlahan naik dimulai pada usia 30 dan diabetes lebih banyak terjadi pada usia 60 keatas.
    
    - *Hypertension* dan *Diabetes*

        Pasien yang tidak mengalami hipertensi lebih banyak mengalami diabetes dari pada pasien yang mengalami hipertensi.
    
    - *Heart Disease* dan *Diabetes*

        Pasien yang tidak memiliki penyakit jantung lebih banyak mengalami diabetes dari pada pasien yang memiliki penyakit jantung.
    
    - *Smoking History* dan *Diabetes*

        Pasien yang mengalami diabetes sebagian besar yang tidak pernah merokok dan mantan perokok.

    - *Body Mass Index* dan *Diabetes*

        Dengan meningkatnya BMI, peluang untuk terkena diabetes akan meningkat.
    
    - *HbA1c Level* dan *Diabetes*

        Dengan meningkat HbA1c Level, peluang untuk terkena diabetes akan meningkat. rata-rata pasien yang mengalami diabetes ketika HbA1c Level sebesar 6 keatas.
    
    - *Blood Glucose Level* dan *Diabetes*
    
        Dengan meningkat Blood Glucose Level, peluang untuk terkena diabetes akan meningkat. rata-rata pasien yang mengalami diabetes ketika Blood Glucose Level sebesar 150 keatas.


5. *Multivariate Analysis*

    Analisis multivariat merujuk pada kumpulan teknik statistik yang digunakan untuk menganalisis dan memahami hubungan antara beberapa fitur secara simultan. Berbeda dengan analisis univariat (yang berfokus pada satu fitur) atau analisis bivariat (yang mempelajari hubungan antara dua fitur), analisis multivariat mempertimbangkan interaksi dan ketergantungan antara tiga atau lebih fitur. Dalam kasus ini, analisis multivariate digunakan seberapa besar korelasi semua fitur terhadap fitur diabetes agar fitur yang memiliki korelasi yang kecil dapat disingkirkan.

    Dari hasil analisis, fitur ``heart_disease`` memiliki korelasi paling kecil. Selanjutnya fitur tersebut di-*drop*. 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Berikut persiapan data yang dilakukan yaitu:

1. Encoding Fitur Kategori

    Setelah melakukan analisis, tersisa beberapa fitur kategorikal dan numerik. fitur kategorikal perlu diubah ke fitur numerik agar model dapat memperlakukan setiap kategori secara terpisah dan tidak memberikan nilai peringkat atau urutan yang tidak relevan. Dengan cara ini, kita dapat menggunakan data kategori dalam model *machine learning* atau analisis data dengan lebih efektif.

    Fitur `gender` dan `smoking_history` adalah fitur kategorikal. Maka perlu diubah ke fitur numerik dengan fungsi pandas `get_dummies()`. Setelah berubah menjadi fitur numerik lalu drop fitur kategorikal sebelumnya. Alhasil semua fitur adalah fitur numerik.

    *One-hot encoding* perlu dilakukan karena banyak algoritma pembelajaran mesin dan analisis data hanya dapat bekerja dengan data numerik. Mereka tidak dapat langsung memproses atau memahami data kategorikal. Jika data kategorikal dianggap sebagai fitur numerik, ini menghilangkan informasi tentang perbedaan antara kategori. Misalnya, jika ada kolom "jenis kelamin" dengan nilai "pria" dan "wanita", dan kita memberikan angka 0 dan 1 masing-masing, ini tidak mencerminkan perbedaan yang sebenarnya antara kategori tersebut. Ini dapat memberikan kesan bahwa ada hubungan ordinal di antara kategori tersebut, meskipun sebenarnya tidak ada. Ini bisa menyesatkan algoritma pembelajaran mesin yang mengasumsikan hubungan ordinal.

2. Train-Test-Split

    Setelah semua fitur berupa numerik, kemudian lakukan *train-test-split*. *train-test-split* adalah metode yang digunakan dalam pembelajaran mesin dan statistik untuk membagi data menjadi dua subset yang saling terpisah, yaitu data pelatihan (training data) dan data pengujian (testing data).

    Selanjutnya pilih fitur target atau fitur yang akan dijadikan prediksi yaitu fitur `diabetes` lalu tandai sebagai variabel "**y**". Fitur-fitur selain fitur target ditandai dengan variabel "**X**".

    *Sklearn* sudah menyiapakan fungsi untuk membagi data latih dan data uji yaitu ``train_test_split()``. Lalu lakukan pemisahan dengan fungsi tersebut, maka data tersebut akan terpisah dengan ditandai nama variabel *train* dan *test*.

    Dengan membagi data menjadi subset pelatihan dan pengujian, kita dapat menghindari penilaian yang terlalu optimis dan mendapatkan perkiraan yang lebih realistis tentang seberapa baik model akan berperforma pada data yang tidak pernah dilihat sebelumnya.

3. Standarisasi

    Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.
 
    StandardScaler adalah salah satu teknik standarisasi yang umum digunakan dalam machine learning. Teknik ini berguna untuk mengubah distribusi data menjadi memiliki mean (rerata) 0 dan standar deviasi (simpangan baku) 1. Dengan menggunakan StandardScaler, setiap fitur akan diperlakukan secara independen dan diskala ulang sehingga memiliki skala yang seragam.

    fitur-fitur yang akan dilakukan standarisasi yaitu ``'age', 'bmi', 'HbA1c_level', 'blood_glucose_level'`` agar fitur-fitur tersebut memiliki skala yang seragam.

    Jika fitur-fitur memiliki skala yang berbeda-beda, interpretasi hasil model dapat menjadi sulit. Misalnya, sulit untuk membandingkan bobot atau koefisien antar fitur jika mereka memiliki skala yang berbeda. Juga, interpretasi pengaruh setiap fitur dalam model dapat menjadi lebih rumit jika skala tidak seragam.


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.



