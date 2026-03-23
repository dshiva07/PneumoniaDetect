# PneumoniaDetect

PneumoniaDetect is a Flask-based web app for uploading a chest image, processing it with OpenCV, and showing a simple AI-style prediction in the browser.

## Project Flow

1. The user opens the home page.
2. The user uploads a medical image from the form.
3. Flask saves the uploaded image into `static/uploads/`.
4. OpenCV reads the image and converts it to grayscale.
5. The app calculates the image mean intensity as a simple prediction rule:
   `mean_intensity > 100` -> `Normal`
   `mean_intensity <= 100` -> `Abnormal`
6. The app applies Canny edge detection to generate a processed image.
7. The processed image is saved in `static/uploads/` with a `processed_` prefix.
8. The page displays:
   original uploaded image
   processed image
   AI prediction result

## Current Tech Stack

- Flask
- OpenCV
- NumPy
- HTML

## File Overview

- `app.py`: Flask backend, upload handling, image processing, and prediction logic
- `templates/index.html`: UI for upload and result display
- `static/uploads/`: stores uploaded and processed images
- `requirements.txt`: Python dependencies

## How To Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Flask app:

```bash
python app.py
```

4. Open the local URL shown in the terminal, usually `http://127.0.0.1:5000`.

## Notes

- The current prediction is a simple rule based on grayscale intensity and is not a trained medical AI model.
- This project is a good base for later integrating a real pneumonia classification model from TensorFlow or PyTorch.
