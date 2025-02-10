import pandas as pd
import numpy as np
import logging
import ipaddress
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionPreprocessor:
    def __init__(self, fraud_data: pd.DataFrame, ip_data: pd.DataFrame):
        self.fraud_data = fraud_data.copy()
        self.ip_data = ip_data.copy()
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    def convert_ip_to_int(self):
        """Ensure IP addresses are properly converted to integer format."""
        logging.info("Converting IP addresses to integer format.")

        # Convert float IPs to integers
        self.fraud_data['ip_int'] = self.fraud_data['ip_address'].astype(float).astype(int)

        logging.info("IP address conversion completed successfully.")

    def merge_with_ip_data(self):
        """Merge fraud_data with ip_address_data based on integer IP ranges."""
        logging.info("Merging fraud_data with ip_address_data.")

        # Ensure both ip_int and lower_bound_ip_address are int64
        self.fraud_data["ip_int"] = self.fraud_data["ip_address"].astype(float).astype(np.int64)
        self.ip_data["lower_bound_ip_address"] = self.ip_data["lower_bound_ip_address"].astype(float).astype(np.int64)

        # Sort ip_address_data by lower_bound_ip_address for efficient merging
        self.ip_data = self.ip_data.sort_values("lower_bound_ip_address")

        # Perform merge_asof to match the closest lower bound
        self.fraud_data = pd.merge_asof(
            self.fraud_data.sort_values("ip_int"),
            self.ip_data,
            left_on="ip_int",
            right_on="lower_bound_ip_address",
            direction="backward"
        )

        logging.info("Merging completed successfully.")


    def add_transaction_frequency(self):
        """Calculate transaction frequency per user."""
        logging.info("Adding transaction frequency feature.")
        self.fraud_data['transaction_count'] = self.fraud_data.groupby('user_id')['user_id'].transform('count')
    
    def add_time_based_features(self):
        """Extract hour of day and day of week from purchase_time."""
        logging.info("Adding time-based features.")
        self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
        self.fraud_data['hour_of_day'] = self.fraud_data['purchase_time'].dt.hour
        self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek
    
    def normalize_features(self, columns):
        """Normalize specified numerical features."""
        logging.info("Normalizing numerical features.")
        self.fraud_data[columns] = self.scaler.fit_transform(self.fraud_data[columns])
    
    def encode_categorical_features(self, columns, top_n=10):
        """One-hot encode specified categorical features, keeping only the top N categories."""
        logging.info("Encoding categorical features.")

        for col in columns:
            # Get the most frequent categories
            top_categories = self.fraud_data[col].value_counts().index[:top_n]
            
            # Replace rare categories with "Other"
            self.fraud_data[col] = self.fraud_data[col].apply(lambda x: x if x in top_categories else "Other")

        encoded_data = self.encoder.fit_transform(self.fraud_data[columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(columns))

        self.fraud_data = pd.concat([self.fraud_data, encoded_df], axis=1).drop(columns, axis=1)

    def get_processed_data(self):
        return self.fraud_data

# Example Usage
''' 
    preprocessor = FraudDetectionPreprocessor(fraud_data, ip_address_data)
    preprocessor.convert_ip_to_int()
    preprocessor.merge_with_ip_data()
    preprocessor.add_transaction_frequency()
    preprocessor.add_time_based_features()
    preprocessor.normalize_features(['transaction_count', 'hour_of_day'])
    preprocessor.encode_categorical_features(['country'])
    
    processed_data = preprocessor.get_processed_data()
    logging.info("Preprocessing complete. Processed Data Sample:\n%s", processed_data.head())
'''