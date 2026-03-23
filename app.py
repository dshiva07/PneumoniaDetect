from flask import Flask, render_template, request
import os
import cv2
from werkzeug.utils import secure_filename

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

    if mean_intensity > 100:
        prediction = "Normal"
        print("The image is Normal.")
    else:
        prediction = "Abnormal"
        print("The image is Abnormal.")

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
    )


if __name__ == '__main__':
    app.run(debug=True)
