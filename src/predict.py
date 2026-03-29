import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import run_feature_engineering_pipeline

def load_artifacts():
    """Loads all saved model artifacts."""
    artifacts_path = "models"
    model = joblib.load(os.path.join(artifacts_path, "best_laptop_price_model_final.joblib"))
    scaler = joblib.load(os.path.join(artifacts_path, "scaler.joblib"))
    feature_columns = joblib.load(os.path.join(artifacts_path, "feature_columns.joblib"))
    brand_ratings = joblib.load(os.path.join(artifacts_path, "brand_ratings.joblib"))
    return model, scaler, feature_columns, brand_ratings

def predict_price(laptop_specs):
    """
    Predicts the price for a new laptop specification.
    
    Args:
        laptop_specs (dict or list of dicts): Laptop specifications mimicking the input CSV format.
    
    Returns:
        float or np.array: Predicted price(s) in local currency (INR).
    """
    model, scaler, feature_columns, brand_ratings = load_artifacts()
    
    # Convert input to DataFrame
    if isinstance(laptop_specs, dict):
        df_input = pd.DataFrame([laptop_specs])
    else:
        df_input = pd.DataFrame(laptop_specs)
        
    # Standardize column naming just in case (the pipeline expects these keys)
    # Required keys: 'Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight'
    
    # Run the robust feature engineering pipeline
    df_processed = run_feature_engineering_pipeline(df_input, brand_ratings=brand_ratings)
    
    # Align features with the training set
    # Add missing one-hot columns as 0s
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    # Select and order only the training columns
    X_new = df_processed[feature_columns]
    
    # Scale numerical features
    numeric_cols = scaler.feature_names_in_
    X_new = X_new.astype({col: float for col in numeric_cols})
    X_new.loc[:, numeric_cols] = scaler.transform(X_new[numeric_cols])
    
    # Predict Log Price
    log_predictions = model.predict(X_new)
    
    # Revert Log transformation
    actual_predictions = np.exp(log_predictions)
    
    return actual_predictions

if __name__ == "__main__":
    # Test sample specs
    new_laptop = {
        'Company': 'Apple',
        'TypeName': 'Ultrabook',
        'Inches': 13.3,
        'ScreenResolution': 'IPS Panel Retina Display 2560x1600',
        'Cpu': 'Intel Core i5 2.3GHz',
        'Ram': '8GB',
        'Memory': '256GB SSD',
        'Gpu': 'Intel Iris Plus Graphics 640',
        'OpSys': 'macOS',
        'Weight': '1.37kg'
    }
    
    gaming_laptop = {
        'Company': 'Razer',
        'TypeName': 'Gaming',
        'Inches': 15.6,
        'ScreenResolution': 'Full HD 1920x1080',
        'Cpu': 'Intel Core i7 7700HQ 2.8GHz',
        'Ram': '16GB',
        'Memory': '512GB SSD',
        'Gpu': 'Nvidia GeForce GTX 1060',
        'OpSys': 'Windows 10',
        'Weight': '2.03kg'
    }

    print("Running Prediction for Apple MacBook Pro...")
    apple_price = predict_price(new_laptop)[0]
    print(f"Predicted Price: ₹{apple_price:,.2f}")

    print("\nRunning Prediction for Razer Gaming Laptop...")
    razer_price = predict_price(gaming_laptop)[0]
    print(f"Predicted Price: ₹{razer_price:,.2f}")
