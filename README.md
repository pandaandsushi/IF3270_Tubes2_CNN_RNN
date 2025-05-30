# IF3270_Tubes2_CNN_RNN_LSTM

---

## Deskripsi Singkat

Repositori ini mencakup pelatihan dan evaluasi model klasifikasi menggunakan Convolutional Neural Network (CNN) pada dataset CIFAR-10 serta Recurrent Neural Network (RNN) dan Long Short-Term Memory (LSTM) pada dataset NusaX-Sentiment. Selain menggunakan Keras, forward propagation dari seluruh model juga diimplementasikan secara manual (from scratch) menggunakan NumPy. Evaluasi model dilakukan dengan metrik macro F1-score, serta disertai analisis pengaruh beberapa hyperparameter terhadap performa model.

---

## Cara Setup & Menjalankan Program

### 1. Clone Repository

```bash
git clone https://github.com/pandaandsushi/IF3270_Tubes2_CNN_RNN.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Program

#### CNN

#### Simple RNN

Untuk melatih model menggunakan Keras dan scratch dan membandingkan F1 score, jalankan:
```bash
python src/rnn/train_rnn.py  
```
Untuk melakukan eksperimen dengan beberapa hyperparameter, jalankan:
```bash
python src/rnn/experiment_rnn.py
```
Hasil evaluasi akan disimpan di `src/rnn/experiments/`.

---

#### LSTM

Untuk melatih model menggunakan Keras, jalankan:
```bash
python src/lstm/train_lstm.py  
```
Untuk melakukan eksperimen dengan beberapa hyperparameter, jalankan:
```bash
python src/lstm/experiment_lstm.py
```
Untuk menguji forward propagation from scratch, jalankan:
```bash
python src/lstm/forward_lstm.py
```
Hasil evaluasi akan disimpan di `src/lstm/experiments/` dan `src/lstm`.

---

## Pembagian Tugas Kelompok

**Kelompok 2:**

| Name              | NIM          | Task Description                                                                                                                                             |
|-------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Thea Josephine Halim**    | 13522012      | Simple RNN
| **Raffael Boymian Siahaan**    | 13522046      | CNN
| **Novelya Putri Ramadhani**    | 13522096      | LSTM

---