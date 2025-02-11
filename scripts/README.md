# Fraud Detection in E-commerce and Banking Transactions

## 📌 Project Overview
This project aims to detect fraudulent transactions in e-commerce and banking systems using **Machine Learning (ML)** and **Deep Learning (DL)** models. It covers data preprocessing, model training, evaluation, and MLOps integration for efficient deployment and tracking.

---

## 📂 Folder Structure
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── logs                  # Logs for monitoring model performance
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & modeling
│   ├── Task_1_Data_Analysis_and_Preprocessing.ipynb  # Data exploration & preprocessing
│   ├── Task_2_Fraud_Detection_by_DL_models.ipynb      # Fraud detection using DL models
│   ├── Task_2_Fraud_Detection_by_ML_Models.ipynb      # Fraud detection using ML models
├── scripts               # Python scripts for automation
│   ├── data_load_clean_transform.py       # Load, clean & transform data
│   ├── fraud_detection_preprocessor.py    # Merge, transform & convert data types
│   ├── fraud_detection.py                 # Fraud detection using ML models
│   ├── fraud_detection_dl.py              # Fraud detection using DL models
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies (TensorFlow, Scikit-learn, MLflow, Pandas, etc.)
```

---

## 🔍 Dataset Description
The project uses two datasets:
1. **Fraud Data** (E-commerce Transactions)
   - User details, transaction timestamps, device/browser info, IP address, fraud labels
   - Processed into a `.pkl` file for ML/DL modeling
2. **Credit Card Data** (Bank Transactions)
   - PCA-transformed transaction features (V1–V28), amount, and fraud labels
   - Stored in `.csv` format

---

## 🚀 Features Implemented
✅ **Data Cleaning & Preprocessing** (Merging, type conversion, feature engineering)
✅ **Machine Learning Models** (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting)
✅ **Deep Learning Models** (MLP, CNN, RNN, LSTM using TensorFlow)
✅ **MLOps Integration** (MLflow for experiment tracking)
✅ **Logging & Monitoring** (Automatic log saving in `logs/logs.log`)
✅ **Unit Testing** (Validation checks on processed data)
✅ **Kaggle GPU Support** (Optimized deep learning models for Kaggle)

---

## 🛠 Setup Instructions
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/zereay74/Kaim-week-8-9.git
cd Kaim-week-8-9
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

### 3️⃣ Run Data Preprocessing
```sh
python scripts/data_load_clean_transform.py
python scripts/fraud_detection_preprocessor.py
```

### 4️⃣ Train Machine Learning Models
```sh
python scripts/fraud_detection.py
```

### 5️⃣ Train Deep Learning Models
```sh
python scripts/fraud_detection_dl.py
```

### 6️⃣ Run Tests
```sh
pytest tests/
```

---

## 📊 Model Evaluation
- **Training/Validation Accuracy & Loss Plots**
- **Test Set Metrics: Accuracy, Precision, Recall, F1-score**

---

## ⚡ MLOps with MLflow
- **Track experiments with MLflow**
- **Compare different model performances**

Run MLflow UI:
```sh
mlflow ui
```

---

## 💡 Future Improvements
- Hyperparameter tuning
- Deploy models via FastAPI
- Real-time fraud detection pipeline

---

## 👨‍💻 Contributors
- Your Name [GitHub](https://github.com/zereay74)

---



