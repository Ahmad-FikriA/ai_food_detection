import os
import uuid

import cv2
import pandas as pd
from flask import Flask, render_template, request
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_DIR = os.path.join(BASE_DIR, 'static', 'results')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

model = YOLO(os.path.join(BASE_DIR, 'model', 'best.pt'))

df_nutrition = pd.read_csv(os.path.join(BASE_DIR, 'nutrition_data.csv'))
df_nutrition['name'] = df_nutrition['name'].astype(str).str.strip().str.title()
df_nutrition.set_index('name', inplace=True)


def get_nutrition(food_name, portion_grams=100):
    food_name = food_name.strip().title()
    if food_name not in df_nutrition.index:
        return None
    row = df_nutrition.loc[food_name]
    scale = portion_grams / 100
    return {
        'kalori': round(row['calories'] * scale, 2),
        'protein': round(row['proteins'] * scale, 2),
        'lemak': round(row['fat'] * scale, 2),
        'karbohidrat': round(row['carbohydrate'] * scale, 2),
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def too_large(_error):
    return render_template('index.html', items=[], image=None,
                           error='File is too large (max 10 MB).'), 413


@app.route('/', methods=['GET', 'POST'])
def index():
    detected_items = []
    image_url = None
    error = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            error = 'Please choose an image to upload.'
        elif not allowed_file(file.filename):
            error = 'Unsupported file type. Use JPG, PNG, WEBP or BMP.'
        else:
            ext = file.filename.rsplit('.', 1)[1].lower()
            name = f'{uuid.uuid4().hex}.{ext}'
            upload_path = os.path.join(UPLOAD_DIR, name)
            file.save(upload_path)

            image = cv2.imread(upload_path)
            if image is None:
                os.remove(upload_path)
                error = 'Could not read that image.'
            else:
                result = model.predict(source=image, conf=0.5, iou=0.7, verbose=False)[0]
                result_name = f'{uuid.uuid4().hex}.jpg'
                cv2.imwrite(os.path.join(RESULT_DIR, result_name), result.plot())
                image_url = f'/static/results/{result_name}'

                for box in result.boxes:
                    food_name = result.names[int(box.cls)]
                    detected_items.append({
                        'food': food_name,
                        'confidence': round(float(box.conf) * 100, 2),
                        'nutrition': get_nutrition(food_name),
                    })
                if not detected_items:
                    error = 'No known food detected. Try a clearer photo.'

    return render_template('index.html', items=detected_items, image=image_url, error=error)


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
