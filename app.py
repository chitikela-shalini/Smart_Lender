from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and preprocessors
MODEL_PATH = 'model/best_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODERS_PATH = 'model/label_encoders.pkl'
FEATURES_PATH = 'model/features.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
features = joblib.load(FEATURES_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Convert numeric columns
        numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])
            
        # Encode categorical columns
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])
            
        # Reorder columns to match training
        input_df = input_df[features]
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = "Approved" if prediction[0] == 1 else "Rejected"
        confidence = round(probability * 100 if prediction[0] == 1 else (1 - probability) * 100, 2)
        
        return render_template('result.html', 
                               result=result, 
                               confidence=confidence,
                               data=data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        
        # Preprocessing same as above
        numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])
            
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])
            
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0]),
            'result': "Approved" if prediction[0] == 1 else "Rejected",
            'confidence': float(probability)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)