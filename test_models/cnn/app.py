# app.py
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_image   # ← اینجا import می‌کنیم

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'هیچ فایلی آپلود نشده'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'فایلی انتخاب نشده'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # اینجا از تابع جدید استفاده می‌کنیم
        result = predict_image(img_path=filepath)

        # فایل موقت را پاک می‌کنیم
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)