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
        """Convert IP addresses to integer format, handling missing or invalid values."""
        logging.info("Converting IP addresses to integer format.")

        # Remove decimal part and convert to a large integer type (int64)
        self.fraud_data['ip_address'] = self.fraud_data['ip_address'].astype(str).str.split('.').str[0]
        self.fraud_data['ip_int'] = self.fraud_data['ip_address'].astype(np.int64)

        logging.info("IP address conversion completed successfully.")

    def merge_with_ip_data(self):
        """Merge fraud_data with ip_address_data on IP ranges."""
        logging.info("Merging fraud_data with ip_address_data.")
        self.fraud_data = self.fraud_data.merge(self.ip_data, how='left', left_on='ip_int', right_on='lower_bound_ip_address')
    
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
    
    def encode_categorical_features(self, columns):
        """One-hot encode specified categorical features."""
        logging.info("Encoding categorical features.")
        encoded_data = self.encoder.fit_transform(self.fraud_data[columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(columns))
        self.fraud_data = pd.concat([self.fraud_data, encoded_df], axis=1).drop(columns, axis=1)
    
    def get_processed_data(self):
        return self.fraud_data

''' 
# Example Usage
if __name__ == "__main__":
    # Load example data
    fraud_data = pd.DataFrame({
        'user_id': [1, 2],
        'purchase_time': ['2025-02-07 12:34:56', '2025-02-07 15:00:00'],
        'ip_address': ['192.168.1.1', '10.0.0.1'],
        'class': [0, 1]
    })
    
    ip_address_data = pd.DataFrame({
        'lower_bound_ip_address': [192168101, 1000001],
        'upper_bound_ip_address': [192168199, 1000999],
        'country': ['USA', 'UK']
    })
    
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