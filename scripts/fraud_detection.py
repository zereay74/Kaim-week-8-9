import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionML:
    def __init__(self, fraud_data_path, creditcard_data_path, model_save_path="models"):
        """Initialize and load datasets"""
        self.fraud_data = pd.read_pickle(fraud_data_path)
        self.creditcard_data = pd.read_csv(creditcard_data_path)
        
        self.models = {
            "LogisticRegression": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier()
        }
        self.scaler = StandardScaler()
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

    def prepare_data(self, dataset_type='fraud'):
        """Prepare dataset by separating features and target, encoding categorical features, and scaling numerical features."""
        if dataset_type == 'fraud':
            df = self.fraud_data.copy()
            X = df.drop(columns=['class'])
            y = df['class']
        else:
            df = self.creditcard_data.copy()
            X = df.drop(columns=['Class'])
            y = df['Class']

        # Convert timestamp columns to UNIX timestamp
        if 'signup_time' in X.columns and 'purchase_time' in X.columns:
            X['signup_time'] = pd.to_datetime(X['signup_time']).astype('int64') // 10**9
            X['purchase_time'] = pd.to_datetime(X['purchase_time']).astype('int64') // 10**9

        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns

        # Apply frequency encoding
        for col in categorical_columns:
            freq_encoding = X[col].value_counts().to_dict()
            X[col] = X[col].map(freq_encoding)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Scaling (only on numerical features)
        numeric_columns = X_train.select_dtypes(include=['number']).columns
        X_train[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = self.scaler.transform(X_test[numeric_columns])

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, dataset_type='fraud'):
        """Train and evaluate models with MLflow logging"""
        X_train, X_test, y_train, y_test = self.prepare_data(dataset_type)
        
        mlflow.set_experiment(f"Fraud_Detection_{dataset_type}")
        best_model = None
        best_score = 0
        
        for model_name, model in self.models.items():
            with mlflow.start_run():
                logging.info(f"Training {model_name} on {dataset_type} dataset...")
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                acc = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')
                
                # Log model and metrics to MLflow
                mlflow.log_param("model", model_name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(model, model_name)
                
                logging.info(f"{model_name} Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                logging.info("Classification Report:\n" + classification_report(y_test, predictions))
                
                # Determine the best model based on F1 Score
                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_model_name = model_name
        
        # Save the best model and scaler
        if best_model:
            self.save_best_model(best_model, best_model_name, dataset_type)

    def save_best_model(self, model, model_name, dataset_type):
        """Save the best performing model and scaler using joblib."""
        model_file = os.path.join(self.model_save_path, f"{dataset_type}_{model_name}.pkl")
        scaler_file = os.path.join(self.model_save_path, f"{dataset_type}_scaler.pkl")
        
        joblib.dump(model, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        logging.info(f"Best model saved: {model_file}")
        logging.info(f"Scaler saved: {scaler_file}")
'''
if __name__ == "__main__":
    fraud_ml = FraudDetectionML("preprocessed_fraud_data.pkl", "creditcard_data.csv")
    fraud_ml.train_and_evaluate('fraud')
    fraud_ml.train_and_evaluate('creditcard')
 '''