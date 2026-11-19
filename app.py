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
FOODS = sorted(model.names.values())

df_nutrition = pd.read_csv(os.path.join(BASE_DIR, 'nutrition_data.csv'))
df_nutrition['name'] = df_nutrition['name'].astype(str).str.strip().str.title()
df_nutrition = df_nutrition.set_index('name')

TEXT = {
    'en': {
        'title': 'Food scanner',
        'intro': 'Upload a photo of a meal. The model finds each dish and estimates its nutrition per 100 g.',
        'drop': 'Drop a food photo here',
        'browse': 'or choose a file',
        'formats': 'JPG, PNG, WEBP or BMP, up to 10 MB',
        'detect': 'Detect food',
        'detecting': 'Detecting…',
        'change': 'Choose another photo',
        'result_alt': 'Your photo with detected foods outlined',
        'found': 'Found on the plate',
        'empty': 'Results appear here after you upload a photo.',
        'confidence': 'confidence',
        'kcal': 'kcal',
        'protein': 'Protein',
        'fat': 'Fat',
        'carbs': 'Carbs',
        'per100': 'per 100 g',
        'total': 'Meal total',
        'total_note': 'Assumes 100 g of each detected food.',
        'no_data': 'No nutrition data for this food.',
        'foods_heading': 'Foods it can recognise',
        'foods_note': '29 dishes and fruits, trained with YOLOv8 on the Indonesian Food dataset.',
        'err_missing': 'Choose an image to upload.',
        'err_type': 'That file type is not supported. Use JPG, PNG, WEBP or BMP.',
        'err_read': 'That file could not be read as an image. Try another photo.',
        'err_size': 'That file is larger than 10 MB. Use a smaller photo.',
        'err_none': 'No known food was found. Try a closer, well-lit photo of one of the foods below.',
        'switch': 'Bahasa Indonesia',
    },
    'id': {
        'title': 'Pemindai makanan',
        'intro': 'Unggah foto makanan. Model akan mengenali setiap hidangan dan memperkirakan gizinya per 100 g.',
        'drop': 'Letakkan foto makanan di sini',
        'browse': 'atau pilih berkas',
        'formats': 'JPG, PNG, WEBP atau BMP, maksimal 10 MB',
        'detect': 'Deteksi makanan',
        'detecting': 'Mendeteksi…',
        'change': 'Pilih foto lain',
        'result_alt': 'Foto Anda dengan makanan yang terdeteksi ditandai',
        'found': 'Ditemukan di piring',
        'empty': 'Hasil akan muncul di sini setelah Anda mengunggah foto.',
        'confidence': 'keyakinan',
        'kcal': 'kkal',
        'protein': 'Protein',
        'fat': 'Lemak',
        'carbs': 'Karbohidrat',
        'per100': 'per 100 g',
        'total': 'Total hidangan',
        'total_note': 'Dihitung 100 g untuk setiap makanan yang terdeteksi.',
        'no_data': 'Data gizi untuk makanan ini belum ada.',
        'foods_heading': 'Makanan yang dapat dikenali',
        'foods_note': '29 hidangan dan buah, dilatih dengan YOLOv8 pada dataset Indonesian Food.',
        'err_missing': 'Pilih gambar untuk diunggah.',
        'err_type': 'Jenis berkas tidak didukung. Gunakan JPG, PNG, WEBP atau BMP.',
        'err_read': 'Berkas tidak dapat dibaca sebagai gambar. Coba foto lain.',
        'err_size': 'Berkas lebih dari 10 MB. Gunakan foto yang lebih kecil.',
        'err_none': 'Tidak ada makanan yang dikenali. Coba foto yang lebih dekat dan terang dari salah satu makanan di bawah.',
        'switch': 'English',
    },
}


def get_lang():
    lang = request.values.get('lang', 'en')
    return lang if lang in TEXT else 'en'


def get_nutrition(food_name, portion_grams=100):
    food_name = food_name.strip().title()
    if food_name not in df_nutrition.index:
        return None
    row = df_nutrition.loc[food_name]
    scale = portion_grams / 100
    protein = row['proteins'] * scale
    fat = row['fat'] * scale
    carbs = row['carbohydrate'] * scale
    # Share of energy from each macro (4 / 9 / 4 kcal per gram) for the macro strip.
    energy = protein * 4 + fat * 9 + carbs * 4 or 1
    return {
        'calories': round(row['calories'] * scale),
        'protein': round(protein, 1),
        'fat': round(fat, 1),
        'carbs': round(carbs, 1),
        'share': {
            'protein': round(protein * 4 / energy * 100, 1),
            'fat': round(fat * 9 / energy * 100, 1),
            'carbs': round(carbs * 4 / energy * 100, 1),
        },
    }


def detect(image_bgr):
    """Run the model on a BGR image. Returns (items, annotated BGR image)."""
    result = model.predict(source=image_bgr, conf=0.5, iou=0.7, verbose=False)[0]
    items = []
    for box in result.boxes:
        food_name = result.names[int(box.cls)]
        items.append({
            'food': food_name,
            'confidence': round(float(box.conf) * 100),
            'nutrition': get_nutrition(food_name),
        })
    return items, result.plot()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def render(lang, items=(), image=None, error=None, status=200):
    known = [i['nutrition'] for i in items if i['nutrition']]
    total = None
    if known:
        total = {k: round(sum(n[k] for n in known), 1) for k in ('protein', 'fat', 'carbs')}
        total['calories'] = sum(n['calories'] for n in known)
    return render_template('index.html', t=TEXT[lang], lang=lang, other='id' if lang == 'en' else 'en',
                           foods=FOODS, items=items, image=image, error=error, total=total), status


@app.errorhandler(413)
def too_large(_error):
    return render(get_lang(), error='err_size', status=413)


@app.route('/', methods=['GET', 'POST'])
def index():
    lang = get_lang()
    if request.method == 'GET':
        return render(lang)

    file = request.files.get('image')
    if not file or not file.filename:
        return render(lang, error='err_missing', status=400)
    if not allowed_file(file.filename):
        return render(lang, error='err_type', status=400)

    ext = file.filename.rsplit('.', 1)[1].lower()
    upload_path = os.path.join(UPLOAD_DIR, f'{uuid.uuid4().hex}.{ext}')
    file.save(upload_path)

    image = cv2.imread(upload_path)
    if image is None:
        os.remove(upload_path)
        return render(lang, error='err_read', status=400)

    items, annotated = detect(image)
    result_name = f'{uuid.uuid4().hex}.jpg'
    cv2.imwrite(os.path.join(RESULT_DIR, result_name), annotated)

    return render(lang, items=items, image=f'/static/results/{result_name}',
                  error=None if items else 'err_none')


if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG') == '1')
