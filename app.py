from flask import Flask, render_template, request
import os
import pandas as pd
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model_path = os.path.join('model','best.pt')
model = YOLO(model_path)

# Load and prepare nutrition data
df_nutrition = pd.read_csv('nutrition_data.csv')
df_nutrition['name'] = df_nutrition['name'].astype(str).str.strip().str.title()
df_nutrition.set_index('name', inplace=True)

def get_nutrition(food_name, portion_grams=100):
    food_name = food_name.strip().title()
    if food_name in df_nutrition.index:
        row = df_nutrition.loc[food_name]
        scale = portion_grams / 100
        return {
            'kalori': round(row['calories'] * scale, 2),
            'protein': round(row['proteins'] * scale, 2),
            'lemak': round(row['fat'] * scale, 2),
            'karbohidrat': round(row['carbohydrate'] * scale, 2)
        }
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    detected_items = []
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Run YOLO detection
            results = model.predict(source=image_path, conf=0.5, iou=0.7)
            for r in results:
                names = r.names
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    food_name = names[class_id]
                    nutrition = get_nutrition(food_name)
                    detected_items.append({
                        'food': food_name,
                        'confidence': round(confidence * 100, 2),
                        'nutrition': nutrition
                    })

    return render_template('index.html', items=detected_items, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
