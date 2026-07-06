import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def generate_mock_data(n_samples=1000):
    print("Generating data...")
    np.random.seed(42)
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1500, 10000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, n_samples),
        'LoanAmount': np.random.randint(50, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 84, 60], n_samples),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Logic for loan status
    score = (df['Credit_History'] * 5) + \
            (df['ApplicantIncome'] / 2000) + \
            (df['Education'] == 'Graduate').astype(int) - \
            (df['LoanAmount'] / 100)
            
    df['Loan_Status'] = (score > 3.5).astype(int)
    return df

def train():
    df = generate_mock_data()
    
    # Create model directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
        
    # Save raw data
    df.to_csv('model/loan_data.csv', index=False)
    
    # Preprocessing
    le_dict = {}
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    features = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train_scaled, y_train)
    
    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(model, 'model/best_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(le_dict, 'model/label_encoders.pkl')
    joblib.dump(features, 'model/features.pkl')
    
    print("Training complete! Artifacts saved in 'model/' directory.")

if __name__ == '__main__':
    train()