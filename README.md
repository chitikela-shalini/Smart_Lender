# Smart Lender - AI Loan Prediction System

Smart Lender is a professional machine learning-powered web application designed to predict the creditworthiness of loan applicants. It uses advanced classification algorithms to help financial institutions make data-driven loan approval decisions.

## 🚀 Features
- **Instant Prediction**: Real-time evaluation of loan applications.
- **Multi-Model Integration**: Compares Decision Tree, Random Forest, KNN, and XGBoost.
- **Interactive Dashboard**: Visual analytics of model performance and feature importance.
- **Premium UI**: Modern, responsive design built with HTML5, CSS3, and JavaScript.
- **REST API**: Endpoint available for third-party integrations.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: NumPy, Pandas, Scikit-learn, XGBoost, SciPy
- **Frontend**: HTML5, CSS3 (Tailwind-inspired), JavaScript, Chart.js
- **Visualization**: Matplotlib, Seaborn

## 📋 Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

## ⚙️ Installation & Setup

1. **Extract the project**
   ```bash
   unzip smart_lender.zip
   cd smart_lender
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models**
   Run the training script to generate the best-performing model (XGBoost):
   ```bash
   python train_models.py
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000`

## 📂 Project Structure
- `app.py`: Main Flask application server.
- `train_models.py`: Script for data generation, preprocessing, and model training.
- `model/`: Directory containing saved models (`.pkl`) and dataset.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS, JS, and image assets.
- `requirements.txt`: List of Python dependencies.

## 👥 Team
- **Team Lead**: Yachham Pavan Kalyan
- **Members**: Chetan Chowdary, Budamala Chatrapathi, Anamala Bhuvan Kumar, K S Mahathwik

---
*Developed for Smart Lender Project - 2024*
