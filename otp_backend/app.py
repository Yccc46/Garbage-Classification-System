from flask import Flask, request, jsonify
import os
import random
import time
import requests
import cv2
import numpy as np
import tensorflow as tf
import uuid
import json
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)


# ==== Firebase 初始化 ====
cred = credentials.Certificate(r"C:\Users\ASUS\Documents\GitHub\Garbage-Classification-System\otp_backend\firebase-key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fyp-db-81903.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

# ==== 模型加载相关 ====
IMG_SIZE = 224
MODEL_PATH = os.environ.get("MODEL_PATH", r"C:\Users\ASUS\Documents\GitHub\Garbage-Classification-System\otp_backend\type_model_converted1.h5")
model = tf.keras.models.load_model(MODEL_PATH)
DATASET_PATH = r"C:\Users\ASUS\Desktop\Train Model\dataset"

ITEMS_INCLUDED = {
    "recyclable": "Recyclable waste includes items like plastic bottles, cans, cartons, and paper.",
    "residual": "Residual waste includes food-soiled paper, broken ceramics, and contaminated plastics.",
    "hazardous": "Hazardous waste includes batteries, paints, pesticides, and electronic waste.",
    "kitchen": "Kitchen waste includes food scraps, vegetable peels, tea leaves, and expired leftovers."
}

# 分类名称
# 加载 class_indices
with open("class_indices.json", "r") as f:
    CLASS_INDEX_TO_CLASSNAME = json.load(f)


def is_dark(image):
    """判断图像是否过暗（灰度均值小于某阈值）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 100  # 你可以根据实际图像情况调整这个阈值

def preprocess_image(image):
    if is_dark(image):
        # 暗图增强：CLAHE + Laplacian
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # 正常图：原图直接处理
        img = image

    # Laplacian 锐化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    resized = cv2.resize(laplacian, (IMG_SIZE, IMG_SIZE))
    input_tensor = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # 单通道模型输入
    return input_tensor

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    user_id = request.form.get('userId')
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 预处理
    processed_image = preprocess_image(image)
    input_tensor = processed_image

    # 预测
    prediction = model.predict(input_tensor)
    category_index = np.argmax(prediction[0])
    category = CLASS_INDEX_TO_CLASSNAME[category_index]
    category_id = category  # 如果 category 名称就对应 Category collection 的 doc ID
    description = ITEMS_INCLUDED.get(category, "")

    # 上传到 Firebase Storage
    image_id = str(uuid.uuid4())
    blob = bucket.blob(f'userHistory/{user_id}/{image_id}.jpg')
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    image_url = blob.generate_signed_url(datetime.timedelta(days=730))  # 2年有效期

    # 获取 Category document 的 reference
    category_ref = db.collection('Category').document(category_id)

    # 写入 Firestore，Category 字段使用 Reference 类型
    history_ref = db.collection('User').document(user_id).collection('History').document()
    history_ref.set({
        "Image_URL": image_url,
        "Category": category_ref,
        "Operation_Type": "Image Recognition",
                "Content": description,
        "Stored_Date": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"category": category_id, "imageUrl": image_url,"description": description})


# ==== OTP 功能 ====
otp_storage = {}  # {email: {otp: ..., expiry: ...}}

# Generate 5-digit OTP
def generate_otp():
    return str(random.randint(10000, 99999))

# Send email using Brevo API
def send_email_brevo(to_email, subject, content):
    url = "https://api.brevo.com/v3/smtp/email"
    api_key = os.getenv("BREVO_API_KEY")  
    if not api_key:
        print("ERROR: BREVO_API_KEY is missing!!!")
    else:
        print("BREVO_API_KEY is loaded.")

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }
    data = {
        "sender": {"name": "Garbage Classification System", "email": "chun040406@gmail.com"},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": f"<p>{content}</p>"
    }
    

    try:
        response = requests.post(url, headers=headers, json=data)
        print("Brevo Response:", response.status_code, response.text)
        return response.ok
    except Exception as e:
        print("Email error:", str(e))
        return False

@app.route('/send_otp_email', methods=['POST'])
def request_otp():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'success': False,'error': 'Missing email'}), 400

    otp = generate_otp()
    expiry = int(time.time()) + 300  # 5 分钟有效

    otp_storage[email] = {"otp": otp, "expiry": expiry}

    success = send_email_brevo(
        to_email=email,
        subject="Your OTP Code",
        content=f"Your OTP is: <strong>{otp}</strong>. It expires in 5 minutes."
    )

    if success:
        return jsonify({'success': True, 'message': 'OTP sent successfully'}), 200
    else:
        return jsonify({'success': False, 'error': 'Failed to send OTP'}), 500


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email')
    otp_input = data.get('otp')

    print(f"Verifying OTP for {email} with input {otp_input}")
    print(f"Stored OTPs: {otp_storage}")

    if not email or not otp_input:
        return jsonify({'success': False,'error': 'Missing email or otp'}), 400

    stored = otp_storage.get(email)
    if not stored:
        return jsonify({'success': False,'error': 'No OTP found for this email'}), 404

    if int(time.time()) > stored['expiry']:
        return jsonify({'success': False,'error': 'OTP has expired'}), 400

    if stored['otp'] != otp_input:
        return jsonify({'success': False, 'error': 'Invalid OTP'}), 401


    del otp_storage[email]
    return jsonify({'success': True, 'message': 'OTP verified successfully'}), 200


@app.route('/')
def home():
    return " OTP Flask backend is running on Render!"

def index():
    return '✅ Version 20250728'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
