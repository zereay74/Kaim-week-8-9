import os 
import sys
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

 
# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'logs.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to handle loading data from CSV files.
    """
    
    def __init__(self):
        """
        Initializes the DataLoader.
        """
        pass

    def load_csv(self, file_path):
        """
        Loads the CSV file into a pandas DataFrame.

        :param file_path: str, path to the CSV file
        :return: pd.DataFrame containing the data from the CSV file
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data successfully loaded from {file_path}")
            logger.info(f"DataFrame Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Error: No data in file at {file_path}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading the file: {e}")
            return None
# Example usage:
# loader = DataLoader()
# df = loader.load_data("path_to_your_file.csv")



class DataCleaner:
    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a DataFrame.
        :param dataframe: pandas DataFrame to be cleaned.
        """
        self.df = dataframe

    def check_missing_values(self):
        """
        Check for missing values in each column.
        :return: DataFrame with columns, missing value count, and percentage.
        """
        logger.info("Checking for missing values in the DataFrame.")
        missing_info = self.df.isnull().sum()
        missing_percentage = (missing_info / len(self.df)) * 100
        logger.info("Missing values check completed.")
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Values': missing_info,
            'Missing Percentage': missing_percentage,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)

    def transform_datetime(self, column, timezone):
        """
        Convert a datetime column to a specified timezone.
        :param column: Name of the datetime column.
        :param timezone: Target timezone (e.g., 'UTC', 'America/New_York').
        """
        logger.info(f"Transforming datetime column '{column}' to timezone '{timezone}'.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        try:
            self.df[column] = pd.to_datetime(self.df[column])
            self.df[column] = self.df[column].dt.tz_localize(None).dt.tz_localize(timezone)
            logger.info(f"Datetime transformation for column '{column}' completed.")
        except Exception as e:
            logger.error(f"An error occurred while transforming datetime: {e}")
    def drop_column(self, column):
        """
        Drop a specified column from the DataFrame.
        :param column: Name of the column to drop.
        """
        logger.info(f"Dropping column: {column}.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        self.df.drop(columns=[column], inplace=True)
        logger.info(f"Column '{column}' dropped successfully.")

    def standardize_column_names(self):
        """
        Standardize column names by converting them to lowercase and replacing spaces with underscores.
        """
        logger.info("Standardizing column names.")
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        logger.info("Column names standardized successfully.")

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        logger.info("Removing duplicate rows from the DataFrame.")
        self.df.drop_duplicates(inplace=True)
        logger.info("Duplicate rows removed successfully.")

    def remove_nulls_from_columns(self, columns):
        """
        Remove rows with null values in the specified columns from the DataFrame.
        :param columns: List of column names to check for null values.
        """
        logger.info(f"Removing rows with null values in columns: {columns}.")
        if not isinstance(columns, list):
            logger.error("Columns parameter should be a list of column names.")
            return
        
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            logger.error(f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
            return

        self.df.dropna(subset=columns, inplace=True)
        logger.info(f"Rows with null values in columns {columns} removed successfully.")


    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a DataFrame.
        :param dataframe: pandas DataFrame to be cleaned.
        """
        self.df = dataframe

    def remove_new_lines(self, column):
        """
        Remove new lines from a specified column.
        :param column: Name of the column to process.
        """
        logger.info(f"Removing new lines from column: {column}.")
        if column in self.df.columns:
            self.df[column] = self.df[column].astype(str).replace(r'\n', ' ', regex=True)
            logger.info(f"New lines removed from column: {column}.")
        else:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")

    def convert_column_to_integer(self, column):
        """
        Convert a specified column to an integer.
        :param column: Name of the column to convert.
        """
        logger.info(f"Converting column '{column}' to integer.")
        if column in self.df.columns:
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce').fillna(0).astype(int)
            logger.info(f"Column '{column}' successfully converted to integer.")
        else:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")

    def fill_null_values(self, columns):
        """
        Replace null values (NaN, None, or empty strings) in specified columns with 'no <column> values'.
        :param columns: List of columns to process.
        """
        logger.info(f"Filling null values in columns: {columns}.")
        
        for column in columns:
            if column in self.df.columns:
                # Count actual NaN values
                filled_count = self.df[column].isna().sum()
                
                # Count "nan" as a string
                nan_string_count = (self.df[column].astype(str) == "nan").sum()

                # If any NaN or "nan" string exists, replace them
                if filled_count > 0 or nan_string_count > 0:
                    logger.info(f"Found {filled_count} NaN and {nan_string_count} 'nan' strings in '{column}', replacing them.")

                    # Replace "nan" strings and other missing values
                    self.df[column] = (
                        self.df[column]
                        .replace(["nan", "NaN", "None", "", None, pd.NaT], pd.NA)
                        .fillna(f'no {column} values')
                    )

                    logger.info(f"Successfully replaced missing values in '{column}'.")
                else:
                    logger.info(f"No NaN values found in '{column}'.")
            else:
                logger.error(f"Column '{column}' does not exist in the DataFrame.")

    def remove_emojis(self, column):
        """
        Remove all types of emojis from a specified column.
        :param column: Name of the column to process.
        """
        logger.info(f"Removing emojis from column: {column}.")
        if column in self.df.columns:
            emoji_pattern = re.compile(
                """[\U0001F600-\U0001F64F]  # emoticons
                |[\U0001F300-\U0001F5FF]  # symbols & pictographs
                |[\U0001F680-\U0001F6FF]  # transport & map symbols
                |[\U0001F700-\U0001F77F]  # alchemical symbols
                |[\U0001F780-\U0001F7FF]  # Geometric Shapes Extended
                |[\U0001F800-\U0001F8FF]  # Supplemental Arrows-C
                |[\U0001F900-\U0001F9FF]  # Supplemental Symbols and Pictographs
                |[\U0001FA00-\U0001FA6F]  # Chess Symbols
                |[\U0001FA70-\U0001FAFF]  # Symbols and Pictographs Extended-A
                |[\U00002702-\U000027B0]  # Dingbats
                |[\U000024C2-\U0001F251]  # Enclosed characters
                """ , re.VERBOSE)
            self.df[column] = self.df[column].astype(str).apply(lambda x: emoji_pattern.sub(r'', x))
            logger.info(f"Emojis removed from column: {column}.")
        else:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")

    def extract_youtube_links(self, source_column, new_column):
        """
        Extract YouTube links from a specified column and save to a new column.
        :param source_column: Column containing text with potential YouTube links.
        :param new_column: Column to store extracted YouTube links.
        """
        logger.info(f"Extracting YouTube links from column '{source_column}' to '{new_column}'.")
        youtube_pattern = r'(https?://(?:www\.|m\.)?youtube\.com/watch\?v=[\w-]+|https?://youtu\.be/[\w-]+)'
        if source_column in self.df.columns:
            self.df[new_column] = self.df[source_column].astype(str).apply(lambda x: re.findall(youtube_pattern, x)[0] if re.findall(youtube_pattern, x) else np.nan)
            logger.info(f"YouTube links extracted and stored in column '{new_column}'.")
        else:
            logger.error(f"Column '{source_column}' does not exist in the DataFrame.")

'''

    # Initialize the cleaner
    cleaner = DataCleaner(df)

    # Check missing values
    print("Missing Values:")
    print(cleaner.check_missing_values())

    # Fill missing values
    cleaner.fill_missing_values()

    # Remove duplicates
    cleaner.remove_duplicates()

    # Standardize column names
    cleaner.standardize_column_names()

    # Transform datetime column
    cleaner.transform_datetime('joining_date', 'UTC')

    # Drop a column
    cleaner.drop_column('name')
    cleaner.remove_new_lines('column_name')
    cleaner.convert_column_to_integer('id_column')
    cleaner.fill_null_values(['column1', 'column2'])
    cleaner.remove_emojis('text_column')
    cleaner.extract_youtube_links('source_column', 'youtube_links')

'''