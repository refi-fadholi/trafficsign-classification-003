from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = FastAPI()

# Path ke model .h5 dan model .tflite
H5_MODEL_PATH = 'models/model_base_002.h5'
TFLITE_MODEL_PATH = 'models/model_ht.tflite'
CSV_FILE = 'data/test-preprocessed2.csv'


# Konversi model dari .h5 ke TFLite jika belum ada
if not os.path.exists(TFLITE_MODEL_PATH):
    model = tf.keras.models.load_model(H5_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Path ke file CSV

# Fungsi untuk memuat dan memproses gambar
def load_image(img_path: str, target_size: tuple):
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Image file '{img_path}' not found.")
    img = image.load_img(img_path, target_size=target_size)  # Load gambar dengan ukuran target
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    return img_array

# Fungsi untuk melakukan prediksi menggunakan model TFLite
def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.get("/predict")
async def predict():
    try:
        # Baca CSV file
        if not os.path.exists(CSV_FILE):
            raise HTTPException(status_code=404, detail=f"CSV file '{CSV_FILE}' not found.")
        
        df = pd.read_csv(CSV_FILE)
        
        # Pastikan kolom CSV bernama 'path' dan 'class'
        if 'path' not in df.columns or 'class' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'path' and 'class' columns")
        
        # Path gambar dari kolom 'path'
        image_paths = df['path'].tolist()
        # Kelas gambar dari kolom 'class'
        true_classes = df['class'].tolist()
        
        # Batasi jumlah data untuk diproses
        max_samples = 100
        image_paths = image_paths[:max_samples]
        true_classes = true_classes[:max_samples]
        
        # Load dan preprocess gambar
        target_size = (256, 256)  # Sesuaikan dengan ukuran input model Anda
        images = [load_image(path, target_size) for path in image_paths]
        
        # Konversi gambar menjadi array untuk prediksi
        predictions = [predict_with_tflite(interpreter, img) for img in images]

        # Mengambil dari role_map (File google colab)
        class_labels = {
            0: "larangan-berhenti",
            1: "larangan-masuk-bagi-kendaraan-bermotor-dan-tidak-bermotor",
            2: "larangan-parkir",
            3: "lampu-hijau",
            4: "lampu-kuning",
            6: "larangan-belok-kanan",
            7: "larangan-belok-kiri",
            8: "larangan-berjalan-terus-wajib-berhenti-sesaat",
            9: "larangan-memutar-balik",
            10: "peringatan-alat-pemberi-isyarat-lalu-lintas",
            11: "peringatan-banyak-pejalan-kaki-menggunakan-zebra-cross",
            12: "peringatan-pintu-perlintasan-kereta-api",
            13: "peringatan-simpang-tiga-sisi-kiri",
            14: "peringatan-penegasan-rambu-tambahan",
            15: "perintah-masuk-jalur-kiri",
            16: "perintah-pilihan-memasuki-salah-satu-jalur",
            17: "petunjuk-area-parkir",
            18: "petunjuk-lokasi-pemberhentian-bus",
            19: "petunjuk-lokasi-putar-balik",
            20: "petunjuk-penyeberangan-pejalan-kaki"
          }
        
        # Menentukan label prediksi berdasarkan nilai probabilitas tertinggi
        prediction_labels = [class_labels[np.argmax(pred)] for pred in predictions]
        prediction_list = [pred.tolist() for pred in predictions]
        
        # Mengaitkan prediksi dengan kelas yang benar dan label prediksi
        results = [{'image_path': path, 
                    'true_class': true_class, 
                    'prediction': pred, 
                    'predicted_label': label} 
                   for path, true_class, pred, label in zip(image_paths, true_classes, prediction_list, prediction_labels)]
        
        # Mengembalikan hasil sebagai JSON terformat
        return JSONResponse(content={"results": results}, headers={"Content-Type": "application/json"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
