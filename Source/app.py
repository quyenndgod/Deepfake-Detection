from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from utils import process_video_and_predict
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model 1 lần duy nhất
model = load_model('model/InceptionResNetV2_FF.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', error='No video uploaded')

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    # Dự đoán video
    result = process_video_and_predict(video_path, model)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)