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
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from src.feature_engineering import load_data, run_feature_engineering_pipeline, prepare_datasets
from src.hyperparameter_tuning import tune_models # Import the tuned logic we just updated

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Visual settings
sns.set_theme(style="whitegrid")

def evaluate_simple(models, X_train, X_test, y_train, y_test):
    """Basic evaluation for initial comparison."""
    results = []
    for name, model in models.items():
        start_time = time()
        model.fit(X_train, y_train)
        duration = time() - start_time
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae_price = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        
        results.append({
            "Model": name,
            "R2 Score": r2,
            "MAE (Price)": mae_price,
            "Time (s)": duration
        })
    return pd.DataFrame(results)

def main():
    data_path = os.path.join(PROJ_ROOT, "data", "laptopData.csv")
    spec_path = os.path.join(PROJ_ROOT, "data", "specData.csv")
    report_dir = os.path.join(PROJ_ROOT, "reports")
    results_dir = os.path.join(PROJ_ROOT, "results")
    model_dir = os.path.join(PROJ_ROOT, "models")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Brand Ratings calculation (Matches Notebook Section 1)
    brand_ratings = None
    if os.path.exists(spec_path):
        print("Calculating brand ratings from spec data...")
        spec_df = pd.read_csv(spec_path)
        spec_df['spec_rating'] = pd.to_numeric(spec_df['spec_rating'], errors='coerce')
        spec_df['brand_match'] = spec_df['brand'].astype(str).str.lower().str.strip()
        brand_ratings = spec_df.groupby('brand_match')['spec_rating'].mean().to_dict()

    print("Loading and preprocessing data...")
    df_raw = load_data(data_path)
    df_features = run_feature_engineering_pipeline(df_raw, brand_ratings=brand_ratings).dropna()
    X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_features)

    # Baseline Training (Matches Notebook Section 2)
    print("\n--- Phase 1: Baseline Comparison ---")
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "CatBoost": CatBoostRegressor(n_estimators=100, verbose=0, random_state=42),
        "Stacked Generalizer (Baseline)": StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=100, random_state=42)),
                ('svr', SVR(kernel='rbf', C=10))
            ],
            final_estimator=RidgeCV(),
            cv=5,
            n_jobs=-1
        )
    }

    baseline_df = evaluate_simple(baseline_models, X_train, X_test, y_train, y_test)
    baseline_df = baseline_df.sort_values(by="R2 Score", ascending=False)
    print(baseline_df[["Model", "R2 Score", "MAE (Price)"]])

    # Hyperparameter Tuning & Tuned Ensemble (Matches Notebook Sections 5 & 5b)
    print("\n--- Phase 2: Hyperparameter Tuning Finalists ---")
    best_models, final_df = tune_models(X_train, y_train, X_test, y_test)
    
    print("\n--- Phase 3: Final Tuned Leaderboard ---")
    final_df = final_df.sort_values(by="R2 Score", ascending=False)
    print(final_df[["Model", "R2 Score", "MAE (Price)"]])
    
    # Save statistics
    final_df.to_csv(os.path.join(results_dir, "final_tuned_performance.csv"), index=False)

    # Identification and Production Save (Matches Notebook Section 6)
    best_overall_name = final_df.iloc[0]["Model"]
    model_key = "Stacked Ensemble" if "Ensemble" in best_overall_name else best_overall_name.replace("Tuned ", "")
    best_model = best_models[model_key]

    model_save_path = os.path.join(model_dir, "best_laptop_price_model_final.joblib")
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "feature_columns.joblib"))
    if brand_ratings:
        joblib.dump(brand_ratings, os.path.join(model_dir, "brand_ratings.joblib"))
    
    print(f"\n WINNER: {best_overall_name}")
    print(f"R² Score: {final_df.iloc[0]['R2 Score']:.4f}")
    print(f"Final Model saved to: {model_save_path}")

    # Visual Visualization (Matches Notebook Section 7)
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkviolet')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Log Price")
    plt.ylabel("Predicted Log Price")
    plt.title(f"Production Model Fit: {best_overall_name}")
    plt.savefig(os.path.join(report_dir, "production_model_fit.png"))
    plt.close()
    print(f"Performance plot saved to {report_dir}")

if __name__ == "__main__":
    main()
