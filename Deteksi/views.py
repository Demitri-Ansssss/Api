import os
from PIL import Image  # Perbaikan: Menggunakan PIL untuk manipulasi gambar
from django.http import JsonResponse
import numpy as np
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Pastikan file .tflite berada di lokasi yang benar
MODEL_APP = os.path.join(os.path.dirname(__file__), 'models', 'model_sampah.tflite')

# Inisialisasi TensorFlow Lite Interpreter
model = tf.lite.Interpreter(model_path=MODEL_APP)
model.allocate_tensors()

# Daftar kelas yang diprediksi
class_names = ['plastik', 'kertas', 'organik']

@csrf_exempt
def predict(request):
    img_height = 224  # Target image height
    img_width = 224  # Target image width

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Ambil file gambar dari request
            uploadFile = request.FILES['image']
            image = Image.open(uploadFile)  # Buka gambar menggunakan PIL
            image = image.resize((img_width, img_height))  # Resize gambar ke ukuran target
            img_array = np.array(image, dtype=np.float32) / 255.0  # Normalisasi gambar
            img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

            # Ambil detail input dan output model
            input_details = model.get_input_details()
            output_details = model.get_output_details()

            # Atur tensor input
            model.set_tensor(input_details[0]['index'], img_array)
            model.invoke()  # Jalankan model

            # Ambil hasil prediksi
            predictions = model.get_tensor(output_details[0]['index'])
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]

            return JsonResponse({
                "message": "Detection successful",
                "predicted_class": predicted_class,
                "confidence": 100*np.max(score)
            }, status=201)
        except Exception as e:
            return JsonResponse({
                "message": "Detection failed",
                "error": str(e)
            }, status=400)
    else:
        return JsonResponse({
            "message": "Invalid request. Please send a POST request with an image."
        }, status=400)
