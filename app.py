from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model dan scaler
MODEL_PATH = 'wine_quality_model.joblib'
SCALER_PATH = 'scaler.joblib'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Label kualitas
QUALITY_LABELS = {
    0: "Rendah",
    1: "Sedang", 
    2: "Tinggi"
}

# Deskripsi kualitas
QUALITY_DESCRIPTIONS = {
    0: "Kualitas rendah (quality ≤ 4)",
    1: "Kualitas sedang (quality 5-6)",
    2: "Kualitas tinggi (quality ≥ 7)"
}

@app.route('/')
def index():
    """Halaman utama dengan form input"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Controller untuk prediksi kualitas wine"""
    try:
        # Validasi model tersedia
        if model is None or scaler is None:
            return jsonify({
                'berhasil': False,
                'pesan_error': 'Model belum dimuat. Pastikan file model tersedia.'
            }), 500
        
        # Ambil data dari form
        data = request.get_json()
        
        # Validasi input
        required_fields = [
            'type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'berhasil': False,
                    'pesan_error': f'Kolom {field} tidak ditemukan pada permintaan.'
                }), 400
        
        # Konversi type: red=0, white=1
        wine_type = 0 if data['type'].lower() == 'red' else 1
        
        # Siapkan feature array sesuai urutan training
        features = np.array([[
            wine_type,                          # type (0=red, 1=white)
            float(data['fixed_acidity']),       # fixed acidity
            float(data['volatile_acidity']),     # volatile acidity
            float(data['citric_acid']),         # citric acid
            float(data['residual_sugar']),      # residual sugar
            float(data['chlorides']),           # chlorides
            float(data['free_sulfur_dioxide']), # free sulfur dioxide
            float(data['total_sulfur_dioxide']),# total sulfur dioxide
            float(data['density']),             # density
            float(data['pH']),                  # pH
            float(data['sulphates']),           # sulphates
            float(data['alcohol'])              # alcohol
        ]])
        
        # Scaling data
        features_scaled = scaler.transform(features)
        
        # Prediksi
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Ambil label dan deskripsi
        quality_label = QUALITY_LABELS[prediction]
        quality_desc = QUALITY_DESCRIPTIONS[prediction]
        
        # Format probabilitas
        probabilities = {
            'Rendah': round(prediction_proba[0] * 100, 2),
            'Sedang': round(prediction_proba[1] * 100, 2),
            'Tinggi': round(prediction_proba[2] * 100, 2)
        }
        
        return jsonify({
            'berhasil': True,
            'prediksi': int(prediction),
            'label_kualitas': quality_label,
            'deskripsi_kualitas': quality_desc,
            'probabilitas': probabilities
        })
        
    except ValueError as e:
        return jsonify({
            'berhasil': False,
            'pesan_error': f'Validasi input gagal: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'berhasil': False,
            'pesan_error': f'Prediksi gagal diproses: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'sehat',
        'model_tersedia': model is not None,
        'scaler_tersedia': scaler is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

