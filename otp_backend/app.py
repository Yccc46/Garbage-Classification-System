from flask import Flask, request, jsonify
from otp_service import send_otp_email
import os
from dotenv import load_dotenv



app = Flask(__name__)

@app.route('/send_otp_email', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')

    if not email or not otp:
        return jsonify({'error': 'Missing email or otp'}), 400

    success = send_otp_email(email, otp)
    if success:
        return jsonify({'message': 'OTP sent successfully'}), 200
    else:
        return jsonify({'error': 'Failed to send OTP'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
