from flask import Flask, jsonify, request, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

app = Flask(__name__, static_folder='static', static_url_path='')

# === [1] Firebase Setup ===
# Make sure this file exists in your root (uploaded to Render or local testing)
FIREBASE_KEY_PATH = "precisionagri-3a0e3-firebase-adminsdk-fbsvc-4d014220cd.json"
FIREBASE_DB_URL = "https://precisionagri-3a0e3-default-rtdb.firebaseio.com/"

if not firebase_admin._apps:
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(f"Firebase key file not found at '{FIREBASE_KEY_PATH}'")
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})

# === [2] Load CSV Data and Train Model ===
DATA_PATH = "smart_irrigation_data.csv"  # Replace with your dataset filename

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset CSV not found at '{DATA_PATH}'")

df = pd.read_csv(DATA_PATH)
if 'crop_type' in df.columns:
    label_encoder = LabelEncoder()
    df['crop_type'] = label_encoder.fit_transform(df['crop_type'])

X = df[['soil_moisture', 'humidity', 'temperature', 'crop_type']]
y = df['irrigation_needed']

model = RandomForestClassifier()
model.fit(X, y)

# === [3] Read Live Sensor Data from Firebase ===
def fetch_live_data():
    ref = db.reference('sensor')
    data = ref.get()

    if not data:
        raise ValueError("No live sensor data found in Firebase.")

    # Convert ADC to percentage
    soil_raw = float(data.get("soilMoisture", 0))
    soil_moisture = 100 - ((soil_raw / 4095) * 100)

    return {
        "soil_moisture": round(soil_moisture, 2),
        "temperature": float(data.get("temperature", 0)),
        "humidity": float(data.get("humidity", 0)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# === [4] API Endpoint to Get Prediction ===
@app.route('/api/irrigation', methods=['GET'])
def irrigation_api():
    try:
        crop_type = int(request.args.get("crop_type", 0))  # e.g., 0 = Wheat
        live = fetch_live_data()
        live['crop_type'] = crop_type

        input_df = pd.DataFrame([live])
        pred = model.predict(input_df[['soil_moisture', 'humidity', 'temperature', 'crop_type']])
        live['irrigation_needed'] = int(pred[0])

        return jsonify(live)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === [5] Serve Frontend Page ===
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# === [6] Run App ===
if __name__ == '__main__':
    app.run(debug=True)
