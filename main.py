import os
import pandas as pd
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# --- 1. Set Local Project Base Directory ---
project_base_dir = os.path.abspath('.')  # Use current directory as base
run_folder_name = 'train'

# --- 2. Load YOLOv8 Model ---
best_model_path = os.path.join(project_base_dir, 'runs', 'detect', run_folder_name, 'weights', 'best.pt')

trained_model = None
try:
    trained_model = YOLO(best_model_path)
    print(f"Model YOLOv8 berhasil dimuat dari: {best_model_path}")
except Exception as e:
    print(f"ERROR: Gagal memuat model dari {best_model_path}. Pastikan path benar dan model ada.")
    print(e)

# --- 3. Load and Clean Nutrition Data ---
original_nutrition_data_path = os.path.join(project_base_dir, 'nutrition_data.csv')

nutrition_df = None
try:
    print(f"\nMembaca data nutrisi dari: {original_nutrition_data_path}")
    df_nutrition = pd.read_csv(original_nutrition_data_path)

    print(f"Jumlah baris sebelum filtering: {len(df_nutrition)}")

    # Daftar Makanan yang Diinginkan (HARUS SESUAI DENGAN KELAS MODEL DAN NAMA DI CSV SETELAH CLEANING)
    desired_foods = [
        'Apple', # Ingat, sesuaikan jika di data.yaml Anda 'Apel'
        'Ayam Goreng',
        'Bakso',
        'Banana',
        'Burger',
        'Capcay',
        'Chocolate Chip Cookie',
        'Donat',
        'Ikan Goreng',
        'Kentang Goreng',
        'Kiwi',
        'Mie Goreng',
        'Nasi Goreng',
        'Nasi Putih',
        'Nugget',
        'Pempek',
        'Pineapples',
        'Pizza',
        'Rendang Sapi',
        'Sate',
        'Spaghetti',
        'Steak',
        'Strawberry',
        'Tahu Goreng',
        'Telur Goreng',
        'Telur Rebus',
        'Tempe Goreng',
        'Terong Balado',
        'Tumis Kangkung'
    ]

    # --- Lakukan Cleaning Awal pada Kolom 'name' untuk Konsistensi ---
    if 'name' in df_nutrition.columns:
        # Konversi ke string, hapus spasi di awal/akhir, dan ubah ke Title Case (misal: "nasi goreng" -> "Nasi Goreng")
        df_nutrition['name'] = df_nutrition['name'].astype(str).str.strip().str.title()
        print("Kolom 'name' CSV telah dibersihkan (strip & title case).")

        # Pastikan juga desired_foods memiliki format yang sama
        desired_foods_cleaned = [food.strip().title() for food in desired_foods]
        print(f"Daftar makanan yang diinginkan juga dibersihkan: {desired_foods_cleaned}")

        # Filter DataFrame berdasarkan daftar makanan yang diinginkan yang sudah dibersihkan
        df_cleaned_nutrition = df_nutrition[df_nutrition['name'].isin(desired_foods_cleaned)].copy()
        print(f"Jumlah baris setelah filtering: {len(df_cleaned_nutrition)}")

        # Drop kolom yang tidak diperlukan ('id', 'image')
        columns_to_drop = ['id', 'image']
        df_cleaned_nutrition.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        print("Kolom 'id' dan 'image' berhasil dihapus.")

        # Set kolom 'name' sebagai index
        df_cleaned_nutrition.set_index('name', inplace=True)
        nutrition_df = df_cleaned_nutrition # Assign ke variabel global nutrition_df
        print("Kolom 'name' berhasil dijadikan index.")

        # Periksa apakah ada makanan yang diinginkan tidak ditemukan di CSV setelah filtering
        found_foods = df_cleaned_nutrition.index.tolist() # Gunakan index karena 'name' sudah jadi index
        missing_foods = [food for food in desired_foods_cleaned if food not in found_foods]
        if missing_foods:
            print("\nPeringatan: Makanan berikut ada di daftar 'desired_foods' tetapi tidak ditemukan di CSV setelah cleaning:")
            for food in missing_foods:
                print(f"- {food}")
        else:
            print("\nSemua makanan yang diinginkan ditemukan di CSV.")

        print("\nContoh data nutrisi yang sudah bersih dan siap digunakan:")
        print(nutrition_df.head())

    else:
        print("ERROR: Kolom 'name' tidak ditemukan di CSV. Tidak dapat melakukan filtering dan pemrosesan.")

except FileNotFoundError:
    print(f"ERROR: File '{original_nutrition_data_path}' tidak ditemukan. Pastikan path dan nama file benar.")
except Exception as e:
    print(f"ERROR: Terjadi kesalahan saat memproses file nutrisi: {e}")

# --- 4. Fungsi untuk Mendapatkan Nutrisi Berdasarkan Nama Makanan ---
def get_nutrition_info(food_name_from_model, portion_grams=None):
    """
    Mengambil informasi nutrisi dari DataFrame nutrisi.
    Jika portion_grams diberikan, akan menghitung nutrisi untuk porsi tersebut.
    Jika tidak, akan menampilkan per 100g.
    """
    if nutrition_df is None:
        print("ERROR: Data nutrisi tidak dimuat, tidak dapat mencari nutrisi.")
        return None

    # Normalisasi nama makanan dari model agar sesuai dengan index di nutrition_df
    # Pastikan ini sesuai dengan cleaning yang Anda lakukan di CSV (strip & title())
    food_name_for_lookup = food_name_from_model.strip().title()

    if food_name_for_lookup in nutrition_df.index:
        nutrition = nutrition_df.loc[food_name_for_lookup].copy()

        # Nama kolom nutrisi sesuai dengan CSV Anda
        CALORIES_COL = 'calories'
        PROTEINS_COL = 'proteins'
        FAT_COL = 'fat'
        CARBOHYDRATE_COL = 'carbohydrate'
        # Jika Anda punya kolom 'satuan_porsi_gram' di CSV, tambahkan di sini
        # PORTION_COL = 'satuan_porsi_gram'

        # Asumsi dasar porsi per 100g jika tidak ada kolom porsi spesifik di CSV
        default_base_portion = 100 # gram

        if portion_grams:
            scale_factor = portion_grams / default_base_portion
            print(f"  --- Perhitungan Nutrisi untuk {food_name_from_model} ({portion_grams} gram) ---")
        else:
            scale_factor = 1 # Untuk menampilkan per 100g
            print(f"  --- Nutrisi untuk {food_name_from_model} (per {default_base_portion} gram) ---")

        # Hitung nutrisi
        nutrition_calc = {
            'kalori': nutrition[CALORIES_COL] * scale_factor,
            'protein': nutrition[PROTEINS_COL] * scale_factor,
            'lemak': nutrition[FAT_COL] * scale_factor,
            'karbohidrat': nutrition[CARBOHYDRATE_COL] * scale_factor
        }

        print(f"  Kalori: {nutrition_calc['kalori']:.2f} kcal")
        print(f"  Protein: {nutrition_calc['protein']:.2f} g")
        print(f"  Lemak: {nutrition_calc['lemak']:.2f} g")
        print(f"  Karbohidrat: {nutrition_calc['karbohidrat']:.2f} g")

        # Mengembalikan dictionary dengan nutrisi yang sudah dihitung
        return nutrition_calc
    else:
        print(f"  Nutrisi untuk '{food_name_from_model}' (setelah disesuaikan menjadi '{food_name_for_lookup}') tidak ditemukan di database.")
        return None

# --- 5. Detection and Nutrition Display Section ---

# Periksa apakah model dan data nutrisi berhasil dimuat sebelum melanjutkan
if trained_model is None:
    print("\nTidak dapat melakukan deteksi karena model gagal dimuat.")
elif nutrition_df is None:
    print("\nTidak dapat menampilkan informasi nutrisi karena data nutrisi gagal dimuat.")
else:
    # Prompt user for local image path
    print("\nMasukkan path gambar makanan yang ingin Anda uji (misal: Indonesian-Food-8/test/images/nama_file.jpg):")
    test_image_path_local = input("Path gambar: ").strip()

    if not os.path.exists(test_image_path_local):
        print(f"Error: Gambar uji tidak ditemukan di {test_image_path_local}. Mohon periksa jalurnya.")
    else:
        print(f"\nMelakukan prediksi pada: {test_image_path_local}")

        # Parameter prediksi
        conf_threshold = 0.5
        iou_threshold = 0.7

        # Lakukan prediksi
        # `save=True` akan menyimpan gambar dengan deteksi yang sudah diplot ke runs/detect/predict/
        results = trained_model.predict(source=test_image_path_local, save=True, conf=conf_threshold, iou=iou_threshold)

        total_calories_overall = 0
        total_protein_overall = 0
        total_fat_overall = 0
        total_carbs_overall = 0
        detected_any_food = False

        # Load gambar asli untuk anotasi custom
        original_img = cv2.imread(test_image_path_local)

        if original_img is None:
            print(f"Error: Tidak dapat membaca gambar dari {test_image_path_local}")
        else:
            # Proses setiap deteksi untuk nutrisi dan anotasi
            print("\n--- Detail Nutrisi untuk Setiap Makanan Terdeteksi ---")
            for r in results:
                boxes = r.boxes # Bounding boxes
                names = r.names # Class names (a dictionary mapping class_id to name)

                if len(boxes) == 0:
                    continue # Tidak ada deteksi pada confidence threshold yang diberikan

                for box in boxes:
                    detected_any_food = True
                    class_id = int(box.cls)
                    confidence = float(box.conf)

                    # Koordinat bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detected_food_name = names[class_id]
                    print(f"\n--- Deteksi: {detected_food_name} (Kepercayaan: {confidence:.2f}) ---")

                    # Gambar rectangle dan label pada gambar
                    label = f"{detected_food_name} {confidence:.2f}"

                    # Draw rectangle dengan warna cyan/turquoise (BGR): (255, 255, 0)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 255, 0), 3)

                    # Background untuk text label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(original_img, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (255, 255, 0), -1)

                    # Draw label dengan warna hitam di atas background cyan
                    cv2.putText(original_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 0), 2)

                    # Panggil fungsi nutrisi (asumsi 100g untuk porsi default)
                    # Anda bisa mengganti 'portion_grams=100' dengan estimasi porsi jika ada
                    nutrition_info = get_nutrition_info(detected_food_name, portion_grams=100) # Asumsi 100 gram per deteksi

                    if nutrition_info: # Jika nutrisi ditemukan dan bukan None
                        total_calories_overall += nutrition_info.get('kalori', 0)
                        total_protein_overall += nutrition_info.get('protein', 0)
                        total_fat_overall += nutrition_info.get('lemak', 0)
                        total_carbs_overall += nutrition_info.get('karbohidrat', 0)

                if not detected_any_food:
                    print("Tidak ada objek yang terdeteksi dalam gambar ini pada confidence threshold yang diberikan.")
                else:
                    # Tampilkan gambar hasil deteksi dengan confidence
                    print(f"\n--- Menampilkan Hasil Deteksi dengan Confidence ---")
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.title(f"Hasil Deteksi dengan Confidence {confidence:.2f})")
                    plt.show()