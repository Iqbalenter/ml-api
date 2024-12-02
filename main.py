import os
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
from PIL import Image
from google.cloud import firestore
from fastapi import FastAPI, Response, UploadFile, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials, storage
import jwt
import uuid

# Secret key untuk JWT
SECRET_KEY = "3f5b2e8c1d9f4a6b7e2c5d8f1a3b6e9c2d7f4b1e5a8c3d6f9b2e7"

# Security schema untuk Bearer Token
security = HTTPBearer()

# Fungsi untuk memverifikasi token JWT
def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token telah kedaluwarsa")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token tidak valid")

# Dependency untuk mendapatkan user dari token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    return verify_jwt(token)

# Load model lokal
model_path = './ModelFM2.keras'  # Ganti dengan path lokal model Anda
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully from local file.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize Firebase Admin SDK
cred = credentials.Certificate('./firebase-service-account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'capstone-project-c242-ps030.firebasestorage.app'
})

# Initialize FastAPI
app = FastAPI()

# Inisialisasi Firestore Client
db = firestore.Client.from_service_account_json('./firebase-service-account.json')

# Zona waktu yang diinginkan
timezone = pytz.timezone("Asia/Jakarta")

# Fungsi untuk mendapatkan data makanan dari Firestore
def get_food_data_from_firestore(food_name):
    try:
        food_ref = db.collection('makanan').document(food_name)
        food_doc = food_ref.get()
        if food_doc.exists:
            return food_doc.to_dict()
        else:
            return None
    except Exception as e:
        raise ValueError(f"Error fetching food data from Firestore: {e}")

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    try:
        img = Image.open(image).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

# Fungsi untuk menyimpan gambar ke Firebase Storage dan mendapatkan URL publik
def upload_image_to_firebase(image_file, user_id):
    try:
        image_file.file.seek(0)
        bucket = storage.bucket()
        filename = f"prediction/{user_id}_{uuid.uuid4()}_{image_file.filename}"
        blob = bucket.blob(filename)
        blob.upload_from_file(image_file.file, content_type=image_file.content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        raise ValueError(f"Error uploading image to Firebase Storage: {e}")

# Fungsi untuk menyimpan data hasil prediksi ke dalam dokumen user
def store_prediction_to_user(user_id, prediction_data):
    try:
        user_ref = db.collection('user').document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists:
            raise ValueError(f"User dengan ID {user_id} tidak ditemukan")
        prediction_ref = user_ref.collection('predictions').add(prediction_data)
        print(f"Prediction berhasil disimpan dengan ID: {prediction_ref[1].id}")
    except Exception as e:
        raise ValueError(f"Error storing prediction: {e}")

# Endpoint untuk prediksi gambar
@app.post("/predict_image")
async def predict_image(
    user: dict = Depends(get_current_user),
    user_id: str = Form(...),
    uploaded_file: UploadFile = None,
    response: Response = None
):
    if user.get("id") != user_id:
        response.status_code = 401
        return {"code": 401, "status": "error", "data": {"message": "User ID tidak sesuai dengan token"}}

    if uploaded_file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/bmp"]:
        response.status_code = 400
        return {"code": 400, "status": "error", "data": {"message": "Invalid image format"}}

    try:
        image_data = preprocess_image(uploaded_file.file)
        prediction = model.predict(image_data)
        class_labels = ["bakso", "bubur_ayam", "lontong_balap", "martabak_telur",
                        "nasi_goreng", "pempek", "rawon", "Telur_Balado", "edamame",
                        "french_fries", "hamburger", "hot_dog", "pancakes", "sashimi",
                        "steak", "sushi", "takoyaki"]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        if confidence < 0.70:
            response.status_code = 400
            return {"code": 400, "status": "error", "data": {"message": "Low confidence, try another image"}}

        food_info = get_food_data_from_firestore(predicted_class)
        if not food_info:
            response.status_code = 400
            return {"code": 400, "status": "error", "data": {"message": "Food data not found"}}

        image_url = upload_image_to_firebase(uploaded_file, user_id)
        prediction_data = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "nutritional_info": food_info,
            "image_url": image_url,
            "createdAt": datetime.now(timezone)
        }

        store_prediction_to_user(user_id, prediction_data)

        return {"code": 200, "status": "success", "data": prediction_data}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"code": 500, "status": "error", "data": {"message": "Internal Server Error"}}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
