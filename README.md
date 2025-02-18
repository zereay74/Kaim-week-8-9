# Fraud Detection System

## Overview
This project is a **fraud detection system** designed to identify fraudulent transactions in e-commerce and banking systems. It includes data preprocessing, machine learning (ML) and deep learning (DL) model development, explainability analysis, API deployment, and a web-based dashboard for fraud monitoring.

## **Project Structure**
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── fraud_dashboard       # Task 5 - Build a Dashboard with Flask and Dash
│   ├── README.md
│   ├── app.py
│   ├── fraud_api.py
│   ├── requirements.txt
├── fraud_detection_api   # Task 4 - Model Deployment and API Development
│   ├── logs
│   │   ├── logs.log
│   ├── Dockerfile
│   ├── serve_model.py
│   ├── requirements.txt
├── logs                  # Logs for monitoring model performance
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & modeling
│   ├── Task_1_Data_Analysis_and_Preprocessing.ipynb  # Data exploration & preprocessing
│   ├── Task_2_Fraud_Detection_by_DL_models.ipynb      # Fraud detection using DL models
│   ├── Task_2_Fraud_Detection_by_ML_Models.ipynb      # Fraud detection using ML models
│   ├── Task_3_Model_Explanability.ipynb
├── scripts               # Python scripts for automation
│   ├── data_load_clean_transform.py       # Load, clean & transform data
│   ├── fraud_detection_preprocessor.py    # Merge, transform & convert data types
│   ├── fraud_detection.py                 # Fraud detection using ML models
│   ├── fraud_detection_dl.py              # Fraud detection using DL models
│   ├── model_explainability.py            # Model explainability
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies (TensorFlow, Scikit-learn, MLflow, Pandas, etc.)
```

---
## **Tasks Performed**

### **Task 1: Data Analysis and Preprocessing**
📌 **Goal:** Prepare and clean data for fraud detection models.

✔️ Loaded transaction data from various sources.
✔️ Merged datasets, handled missing values, and removed duplicates.
✔️ Feature engineering: Created new fraud-related features.
✔️ Standardized and normalized numerical features.
✔️ Saved cleaned datasets for model training.

**Files:**
- 📄 `notebooks/Task_1_Data_Analysis_and_Preprocessing.ipynb`
- 📝 `scripts/data_load_clean_transform.py`
- 📝 `scripts/fraud_detection_preprocessor.py`

---
### **Task 2: Model Building and Training**
📌 **Goal:** Train ML & DL models for fraud detection.

✔️ Trained various models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP, CNN, RNN, LSTM.
✔️ Evaluated models using accuracy, precision, recall, and F1-score.
✔️ Saved the best-performing models for deployment.
✔️ Tracked experiments using MLflow.

**Files:**
- 📄 `notebooks/Task_2_Fraud_Detection_by_ML_Models.ipynb`
- 📄 `notebooks/Task_2_Fraud_Detection_by_DL_models.ipynb`
- 📝 `scripts/fraud_detection.py`
- 📝 `scripts/fraud_detection_dl.py`

---
### **Task 3: Model Explainability**
📌 **Goal:** Interpret and explain fraud detection models.

✔️ Used SHAP and LIME for feature importance analysis.
✔️ Visualized model decisions to increase transparency.
✔️ Identified key fraud indicators.

**Files:**
- 📄 `notebooks/Task_3_Model_Explanability.ipynb`
- 📝 `scripts/model_explainability.py`

---
### **Task 4: Model Deployment and API Development**
📌 **Goal:** Serve fraud detection models as an API.

✔️ Developed a **Flask API** to expose model predictions.
✔️ Integrated SQL database for logging detected frauds.
✔️ Dockerized API for scalable deployment.
✔️ Implemented logging for monitoring requests.

**API Endpoints:**
- `/predict`: Detects fraud in new transactions.
- `/fraud_trends`: Returns fraud statistics over time.
- `/fraud_by_country`: Provides fraud data per country.
- `/fraud_by_device_browser`: Summarizes fraud cases per device and browser.

**Files:**
- 📄 `fraud_detection_api/serve_model.py`
- 📄 `fraud_detection_api/Dockerfile`
- 📄 `fraud_detection_api/requirements.txt`
- 📄 `logs/logs.log`

---
### **Task 5: Build a Dashboard with Flask and Dash**
📌 **Goal:** Create a fraud detection dashboard for visualization.

✔️ Built a **Flask backend** to fetch fraud data.
✔️ Developed a **Dash frontend** for interactive data visualization.
✔️ Integrated **Plotly charts**: Line chart, bar chart, and choropleth map.
✔️ Displayed key fraud insights: Total fraud cases, fraud trends, fraud per country/device.
✔️ Improved API error handling and logging.

**Dashboard Features:**
- **Summary Statistics:** Total transactions, fraud cases, fraud percentage.
- **Line Chart:** Fraud trends over time.
- **Bar Chart:** Fraud cases across different devices and browsers.
- **Choropleth Map:** Fraud cases per country.

**Files:**
- 📄 `fraud_dashboard/app.py`
- 📄 `fraud_dashboard/fraud_api.py`
- 📄 `fraud_dashboard/requirements.txt`

---
## **Installation & Usage**

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Fraud Detection API**
```bash
cd fraud_detection_api
python serve_model.py
```

### **3️⃣ Start the Dashboard**
```bash
cd fraud_dashboard
python app.py
```

### **4️⃣ Open in Browser**
```
http://127.0.0.1:8050
```

---
## **Technologies Used**
✔️ **Python** (Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, SHAP, LIME)  
✔️ **ML & DL Models** (Logistic Regression, Decision Tree, CNN, LSTM)  
✔️ **Flask & Dash** (REST API & Interactive Dashboard)  
✔️ **Docker** (Containerized Deployment)  
✔️ **PostgreSQL** (Database for storing fraud data)  
✔️ **MLflow** (Experiment Tracking)  
✔️ **Plotly** (Data Visualization)  

---
## **Contributions**
🚀 Feel free to contribute by submitting issues, feature requests, or pull requests! 🎯

---
## **License**
This project is licensed under the **MIT License**.

---
📢 **Follow for Updates!** 🚀🔥

