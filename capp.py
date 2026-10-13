
from flask import Flask, request, render_template
from pathlib import Path
import pickle
import numpy as np

app = Flask(__name__)

# Get the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Load trained model and scaler
with open(BASE_DIR / "rf_classifier.pkl", "rb") as model_file:
    rf_classifier = pickle.load(model_file)

with open(BASE_DIR / "std.pkl", "rb") as scaler_file:
    standard_scaler = pickle.load(scaler_file)

# Crop dictionary
crop = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton',
    5: 'coconut', 6: 'papaya', 7: 'orange',
    8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana',
    14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean',
    18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea',
    22: 'coffee'
}

# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':
        try:
            N = float(request.form.get('N'))
            P = float(request.form.get('P'))
            K = float(request.form.get('K'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            rainfall = float(request.form.get('rainfall'))
            ph = float(request.form.get('ph'))

            # Prepare input features
            features = np.array([[
                N, P, K, temperature,
                humidity, rainfall, ph
            ]])

            # Scale input
            new_data_scaled = standard_scaler.transform(features)

            # Predict crop
            result = rf_classifier.predict(new_data_scaled)

            # Get crop name
            crop_result = crop.get(
                int(result[0]), "Unknown Crop"
            )

            return render_template(
                'home.html',
                result=crop_result
            )

        except (ValueError, TypeError):
            return render_template(
                'home.html',
                result="Error: Please enter valid numeric values."
            ), 400

        except Exception:
            app.logger.exception("Prediction failed")
            return render_template(
                'home.html',
                result="Error: Prediction failed. Please try again."
            ), 500

    return render_template('home.html')


# Run locally
if __name__ == "__main__":
    app.run(debug=False)
