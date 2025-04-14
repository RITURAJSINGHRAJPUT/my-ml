from flask import Flask, jsonify, request, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path='', static_folder='static')

# === Firebase Setup ===
cred_path = "precisionagri-3a0e3-firebase-adminsdk-fbsvc-4d014220cd.json"
db_url = "https://precisionagri-3a0e3-default-rtdb.firebaseio.com/"

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {'databaseURL': db_url})

# === Load & Train Model ===
df = pd.read_csv("your_dataset.csv")
label_encoder = LabelEncoder()
df['crop_type'] = label_encoder.fit_transform(df['crop_type'])

X = df[['soil_moisture', 'humidity', 'temperature', 'crop_type']]
y = df['irrigation_needed']

model = RandomForestClassifier()
model.fit(X, y)

# === Firebase Reading ===
def fetch_live_data():
    ref = db.reference('sensor')
    data = ref.get()

    if not data:
        raise ValueError("No sensor data found")

    soil_moisture = 100 - ((float(data.get("soilMoisture", 0)) / 4095) * 100)

    return {
        "soil_moisture": round(soil_moisture, 2),
        "temperature": float(data.get("temperature", 0)),
        "humidity": float(data.get("humidity", 0)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# === API Endpoint ===
@app.route('/api/irrigation', methods=['GET'])
def irrigation_api():
    try:
        crop_type = int(request.args.get("crop_type", 0))  # default crop_type = 0
        reading = fetch_live_data()
        reading['crop_type'] = crop_type

        input_df = pd.DataFrame([reading])
        pred = model.predict(input_df[['soil_moisture', 'humidity', 'temperature', 'crop_type']])
        reading['irrigation_needed'] = int(pred[0])
        return jsonify(reading)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Serve Frontend ===
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
