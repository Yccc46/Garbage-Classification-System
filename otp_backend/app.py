from flask import Flask, request, jsonify
from flask_cors import CORS
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
import tempfile
from firebase_admin import credentials, firestore, storage
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# ==== Firebase 初始化 ====
# 读取环境变量
firebase_key_str = os.environ.get("FIREBASE_KEY")

# 将字符串写入一个临时文件
with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_key_file:
    temp_key_file.write(firebase_key_str)
    temp_key_path = temp_key_file.name

cred = credentials.Certificate(temp_key_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fyp-db-81903.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

# ==== 模型加载相关 ====
IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"final_model.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


ITEMS_INCLUDED = {
    "recyclable": "Recyclable waste includes items like plastic bottles, cans, cartons, and paper.",
    "residual": "Residual waste includes food-soiled paper, broken ceramics, and contaminated plastics.",
    "hazardous": "Hazardous waste includes batteries, paints, pesticides, and electronic waste.",
    "kitchen": "Kitchen waste includes food scraps, vegetable peels, tea leaves, and expired leftovers."
}

# 分类名称
# 加载 class_indices
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "class_indices.json")

with open(CLASS_INDEX_PATH, "r") as f:
    CLASS_INDEX_TO_CLASSNAME = json.load(f)
    
CLASS_INDEX_TO_CLASSNAME = {int(k): v for k, v in CLASS_INDEX_TO_CLASSNAME.items()}

def apply_clahe_laplacian(img):
      img = np.uint8(img * 255)  # 将归一化图像还原
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # keras 默认是 RGB
      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)

      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      cl = clahe.apply(l)

      merged = cv2.merge((cl, a, b))
      enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

      laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
      laplacian = cv2.convertScaleAbs(laplacian)
      combined = cv2.addWeighted(enhanced, 0.8, laplacian, 0.2, 0)

      combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
      return combined_rgb / 255.0  # 返回归一化后的 RGB 图像


def preprocess_image(image):
    # Resize
    img_array = cv2.resize(image,(224,224))
    
    # CLAHE + Laplacian
    img_array = apply_clahe_laplacian(img_array)

    return img_array.astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    user_id = request.form.get('userId')
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Preprocessing
    processed_image = preprocess_image(image)
    input_tensor = np.expand_dims(processed_image, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    category_index = np.argmax(prediction[0])
    category = CLASS_INDEX_TO_CLASSNAME[category_index]
    category_id = category  
    description = ITEMS_INCLUDED.get(category, "")

    # Upload to Firebase Storage
    image_id = str(uuid.uuid4())
    blob = bucket.blob(f'userHistory/{user_id}/{image_id}.jpg')
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    # 设置为公开访问
    blob.make_public()

    # 获取公开 URL
    image_url = blob.public_url

    # Get Category document reference
    category_ref = db.collection('Category').document(category_id)

    # Write Firestore，Category as Reference type
    history_ref = db.collection('User').document(user_id).collection('History').document()
    history_ref.set({
        "Image_URL": image_url,
        "Category": category_ref,
        "Operation_Type": "Image Recognition",
                "Content": description,
        "Stored_Date": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"category": category_id, "imageUrl": image_url,"description": description})


# ==== OTP Function ====
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
    expiry = int(time.time()) + 300  # valid in 5 minutes

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
