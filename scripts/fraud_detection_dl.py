import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Enable MLflow Auto-Logging
mlflow.tensorflow.autolog()

# Enable GPU Usage on Kaggle
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
logging.info(f"Using {device}")

class FraudDetectionDL:
    def __init__(self, fraud_data_path="/kaggle/input/fraud-data/fraud_detection_preprocessed.pkl", creditcard_data_path="/kaggle/input/fraud-data/creditcard.csv"):
        """Initialize and load data."""
        self.fraud_data = pd.read_pickle(fraud_data_path)
        self.creditcard_data = pd.read_csv(creditcard_data_path)
        self.scaler = StandardScaler()
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

    def build_mlp(self, input_shape):
        """Build a simple Multi-Layer Perceptron (MLP)."""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_cnn(self, input_shape):
        """Build a Convolutional Neural Network (CNN)."""
        model = keras.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),  # Reshape for CNN
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_rnn(self, input_shape):
        """Build a simple Recurrent Neural Network (RNN)."""
        model = keras.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),  # Reshape for RNN
            layers.SimpleRNN(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_lstm(self, input_shape):
        """Build an LSTM network."""
        model = keras.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),  # Reshape for LSTM
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.LSTM(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_and_evaluate(self, model_type='mlp', dataset_type='fraud', epochs=10, batch_size=64):
        """Train and evaluate a deep learning model with MLflow tracking."""
        X_train, X_test, y_train, y_test = self.prepare_data(dataset_type)
        input_shape = X_train.shape[1]

        # Choose the model
        if model_type == 'mlp':
            model = self.build_mlp(input_shape)
        elif model_type == 'cnn':
            model = self.build_cnn(input_shape)
        elif model_type == 'rnn':
            model = self.build_rnn(input_shape)
        elif model_type == 'lstm':
            model = self.build_lstm(input_shape)
        else:
            raise ValueError("Invalid model_type. Choose from 'mlp', 'cnn', 'rnn', or 'lstm'.")

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model with MLflow logging
        with mlflow.start_run():
            logging.info(f"Training {model_type.upper()} on {dataset_type} dataset...")
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

            # Evaluate model
            loss, accuracy = model.evaluate(X_test, y_test)
            logging.info(f"{model_type.upper()} Accuracy: {accuracy:.4f}")

            # Log metrics in MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dataset_type", dataset_type)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.tensorflow.log_model(model, artifact_path="model")

        return model, history
    def plot_training_history(self, history, model_type, dataset_type):
        """Plot training and validation accuracy/loss for a given model."""
        plt.figure(figsize=(12, 5))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{model_type.upper()} - {dataset_type} Dataset: Accuracy')
        plt.legend()

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Val Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_type.upper()} - {dataset_type} Dataset: Loss')
        plt.legend()

        plt.show()

    def evaluate_model(self, model, dataset_type):
        """Compute accuracy, precision, recall, and F1-score on the test data."""
        X_train, X_test, y_train, y_test = self.prepare_data(dataset_type)

        # Predictions
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

        # Compute Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)

        # Print Metrics
        print(f"📌 {model.name.upper()} - {dataset_type.upper()} Dataset Metrics:")
        print(f"🔹 Accuracy  : {accuracy:.4f}")
        print(f"🔹 Precision : {precision:.4f}")
        print(f"🔹 Recall    : {recall:.4f}")
        print(f"🔹 F1 Score  : {f1:.4f}")

        return accuracy, precision, recall, f1