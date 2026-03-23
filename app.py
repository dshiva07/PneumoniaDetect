from flask import Flask, render_template, request
import os
import cv2
from werkzeug.utils import secure_filename



import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model.h5')


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#Create folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected Image')

    if not file:
        return render_template('index.html', error='Upload failed')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return render_template('index.html', error='Uploaded file is not a valid image')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()

    img = Image.open(filepath).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction_value  = model.predict(img_array)[0][0]
    if prediction_value > 0.5:
        prediction = "Pneumonia"
        print("The image is Pneumonia.")    
    else:
        prediction = "Normal"
        print("The image is Normal.")
    confidence = prediction_value if prediction == "Pneumonia" else 1 - prediction_value
    print(f"Confidence: {confidence * 100:.2f}%")

    edges = cv2.Canny(gray, 100, 200)
    processed_filename = 'processed_' + filename
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, edges)

    print(f"Processed image saved as {processed_filename} | Prediction: {prediction}")
    return render_template(
        'index.html',
        uploaded=filename,
        processed=processed_filename,
        prediction=prediction,
        confidence=confidence,
    )


if __name__ == '__main__':
    app.run(debug=True)
