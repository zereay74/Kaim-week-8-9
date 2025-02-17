from scripts.fraud_detection import FraudDetectionML  # Import ML model class
import pandas as pd
import shap
import lime
import lime.lime_tabular
import logging
import matplotlib.pyplot as plt
import multiprocessing


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelExplainability:
    def __init__(self, fraud_ml):
        """Initialize with trained models from FraudDetectionML."""
        self.fraud_ml = fraud_ml
        self.models = fraud_ml.models
        self.X_train_fraud, self.X_test_fraud, self.y_train_fraud, self.y_test_fraud = fraud_ml.prepare_data('fraud')
        self.X_train_credit, self.X_test_credit, self.y_train_credit, self.y_test_credit = fraud_ml.prepare_data('creditcard')
        
    def shap_explain(self, dataset_type='fraud'):
        """Generate SHAP explanations (Summary & Dependence plots)"""
        X_train, X_test = (self.X_train_fraud, self.X_test_fraud) if dataset_type == 'fraud' else (self.X_train_credit, self.X_test_credit)
        model = self.models["RandomForest"]  # Using Random Forest for SHAP

        # Use SHAP's TreeExplainer for faster computation on tree models
        explainer = shap.TreeExplainer(model, X_train, feature_perturbation="interventional", check_additivity=False)
        shap_values = explainer.shap_values(X_test)

        # Ensure correct SHAP values selection (for classification tasks)
        if isinstance(shap_values, list):  
            shap_values = shap_values[1]  # Select SHAP values for class 1

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'shap_summary_{dataset_type}.png')
        plt.show()  # Ensure the plot is displayed
        logging.info(f"SHAP Summary Plot saved for {dataset_type}.png")

        # Dependence Plot (Use first feature)
        plt.figure()
        shap.dependence_plot(0, shap_values, X_test, show=False)
        plt.savefig(f'shap_dependence_{dataset_type}.png')
        plt.show()  # Ensure the plot is displayed
        logging.info(f"SHAP Dependence Plot saved for {dataset_type}.png")


    def lime_explain(self, dataset_type='fraud'):
        """Generate optimized LIME explanations."""
        X_train, X_test = (self.X_train_fraud, self.X_test_fraud) if dataset_type == 'fraud' else (self.X_train_credit, self.X_test_credit)
        model = self.models["LogisticRegression"]  # Using Logistic Regression for LIME
    
        # Ensure feature names are preserved
        feature_names = X_train.columns.tolist()
    
        # Convert X_train to NumPy array for LIME
        X_train_np = X_train.to_numpy()
    
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_np,
            feature_names=feature_names,  
            class_names=["Not Fraud", "Fraud"],  # Add class names if available
            mode="classification",
            discretize_continuous=True
        )
    
        idx = 0  # Choose first instance for explanation
        instance = X_test.iloc[idx].to_numpy().reshape(1, -1)  # Convert single row to NumPy
    
        # ✅ Ensure DataFrame input for model.predict_proba
        def model_predict_proba(x):
            x_df = pd.DataFrame(x, columns=feature_names)  # Convert NumPy array back to DataFrame
            return model.predict_proba(x_df)
    
        # Generate LIME explanation
        exp = explainer.explain_instance(instance.flatten(), model_predict_proba, num_features=10)
    
        # Save explanation
        exp.save_to_file(f'lime_explanation_{dataset_type}.html')
        logging.info(f"LIME Explanation saved for {dataset_type} dataset.")

''' 
if __name__ == "__main__":
    fraud_ml = FraudDetectionML("preprocessed_fraud_data.pkl", "creditcard_data.csv")
    explainability = ModelExplainability(fraud_ml)
    explainability.shap_explain('fraud')
    explainability.shap_explain('creditcard')
    explainability.lime_explain('fraud')
    explainability.lime_explain('creditcard')
'''