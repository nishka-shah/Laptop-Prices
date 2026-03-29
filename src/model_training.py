import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import joblib

# Find project root (one level up from src/)
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure project root is in path for robust imports
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

# Import modular pipeline functions
from src.feature_engineering import load_data, run_feature_engineering_pipeline, prepare_datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    median_absolute_error, 
    mean_absolute_percentage_error
)

# Visual settings
sns.set_theme(style="whitegrid")

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time()
        model.fit(X_train, y_train)
        train_time = time() - start_time
        y_pred = model.predict(X_test)
        
        # Metrics
        mse_log = mean_squared_error(y_test, y_pred)
        rmse_log = np.sqrt(mse_log)
        mae_log = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Adjusted R2
        n = X_test.shape[0]
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        medae_log = median_absolute_error(y_test, y_pred)
        mape_log = mean_absolute_percentage_error(y_test, y_pred)

        y_test_exp = np.exp(y_test)
        y_pred_exp = np.exp(y_pred)
        mae_price = mean_absolute_error(y_test_exp, y_pred_exp)
        rmse_price = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
        mape_price = mean_absolute_percentage_error(y_test_exp, y_pred_exp)
        
        results.append({
            "Model": name,
            "R2 Score": r2,
            "Adj R2": adj_r2,
            "MSE (Log)": mse_log,
            "RMSE (Log)": rmse_log,
            "MAE (Log)": mae_log,
            "MedAE (Log)": medae_log,
            "MAPE (Log)": mape_log,
            "MAE (Price)": mae_price,
            "RMSE (Price)": rmse_price,
            "MAPE (Price)": mape_price,
            "Time (s)": train_time
        })
    return pd.DataFrame(results)

def main():
    # Paths relative to project root
    data_path = os.path.join(PROJ_ROOT, "data", "cleaned_laptop_data.csv")
    report_dir = os.path.join(PROJ_ROOT, "reports")
    results_dir = os.path.join(PROJ_ROOT, "results")
    model_dir = os.path.join(PROJ_ROOT, "models")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("Loading data and training baseline models...")
    df_raw = load_data(data_path)
    df_features = run_feature_engineering_pipeline(df_raw).dropna()
    X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_features)

    # Standardize baseline hyperparameters
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)
    results_df = results_df.sort_values(by="R2 Score", ascending=False)
    print("\nBaseline Results:\n", results_df.to_string(index=False))

    # Save results
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(results_dir, "model_performance.csv"), index=False)

    # Save Best Model & Plots
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    
    # Feature Importance Plot
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feat_imp.values[:20], y=feat_imp.index[:20], hue=feat_imp.index[:20], palette="rocket", legend=False)
        plt.title(f"Top 20 Feature Importances - {best_model_name}")
        plt.savefig(os.path.join(report_dir, "feature_importance.png"))
        plt.close() # Close to avoid overlapping
        print(f"Feature importance plot saved to {report_dir}")

    # Final Accuracy Plot (Actual vs Predicted)
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Log Price")
    plt.ylabel("Predicted Log Price")
    plt.title(f"Final Model Fit: {best_model_name} (R² = {results_df.iloc[0]['R2 Score']:.4f})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(report_dir, "best_model_fit.png"))
    plt.close()
    print(f"Accuracy plot saved to {report_dir}/best_model_fit.png")

    model_save_path = os.path.join(model_dir, "best_laptop_price_model.joblib")
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"\nBest baseline model ({best_model_name}) saved to {model_save_path}")

if __name__ == "__main__":
    main()
