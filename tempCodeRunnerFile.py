from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
rf_classifier = pickle.load(open(r"C:\Users\AKHIL\Downloads\SOIL\rf_classifier.pkl", 'rb'))
standard_scaler = pickle.load(open(r"C:\Users\AKHIL\Downloads\SOIL\std.pkl", 'rb'))

# Crop dictionary
crop = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 7: 'orange',
    8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 12: 'mango', 13: 'banana',
    14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans',
    19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
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

            # Scale input
            new_data_scaled = standard_scaler.transform([[N, P, K, temperature, humidity, rainfall, ph]])
            result = rf_classifier.predict(new_data_scaled)

            # Get crop name from prediction
            crop_result = crop.get(result[0], "Unknown Crop")
            return render_template('home.html', result=crop_result)
        except Exception as e:
            return render_template('home.html', result=f"Error: {str(e)}")
    else:
        return render_template('home.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
