from flask import Flask, request, jsonify
import os
import random
import time
import requests

app = Flask(__name__)
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
        return jsonify({'error': 'Missing email'}), 400

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

    if not email or not otp_input:
        return jsonify({'error': 'Missing email or otp'}), 400

    stored = otp_storage.get(email)
    if not stored:
        return jsonify({'error': 'No OTP found for this email'}), 404

    if int(time.time()) > stored['expiry']:
        return jsonify({'error': 'OTP has expired'}), 400

    if stored['otp'] != otp_input:
        return jsonify({'error': 'Invalid OTP'}), 401

    del otp_storage[email]
    return jsonify({'message': 'OTP verified successfully'}), 200

@app.route('/')
def home():
    return " OTP Flask backend is running on Render!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
