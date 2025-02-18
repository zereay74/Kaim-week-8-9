import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# Load fraud data
DATA_PATH = "../week 8-9 data/Data/merged_fraud_data.csv"  # Ensure this path is correct
fraud_df = pd.read_csv(DATA_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running"}), 200

@app.route("/summary", methods=["GET"])
def get_summary():
    """Return summary statistics about fraud cases"""
    total_transactions = int(len(fraud_df))
    fraud_cases = int(fraud_df["class"].sum())
    fraud_percentage = round((fraud_cases / total_transactions) * 100, 2)

    summary = {
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": fraud_percentage,
    }
    return jsonify(summary)

@app.route("/fraud_trends", methods=["GET"])
def get_fraud_trends():
    """Return fraud trends over time"""
    fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])
    fraud_trends = fraud_df.groupby(fraud_df["purchase_time"].dt.date)["class"].sum().reset_index()
    fraud_trends.columns = ["date", "fraud_cases"]

    # Convert 'fraud_cases' column to list of Python integers
    fraud_trends["fraud_cases"] = fraud_trends["fraud_cases"].astype(int)

    return fraud_trends.to_json(orient="records")

@app.route("/fraud_by_country", methods=["GET"])
def fraud_by_country():
    """Returns fraud cases count grouped by country."""
    fraud_counts = fraud_df[fraud_df["class"] == 1].groupby("country").size().reset_index(name="fraud_count")
    return fraud_counts.to_json(orient="records")

@app.route("/fraud_by_device_browser")
def fraud_by_device_browser():
    """Returns fraud cases grouped by device and browser."""
    try:
        fraud_counts = fraud_df[fraud_df['class'] == 1].groupby(['device_id', 'browser']).size().reset_index(name='fraud_count')
        return fraud_counts.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
