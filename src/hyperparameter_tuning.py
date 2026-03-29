import os
import sys
import pandas as pd
import numpy as np
from time import time
import joblib

# Find project root (one level up from src/)
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from src.feature_engineering import load_data, run_feature_engineering_pipeline, prepare_datasets
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

def tune_models(X_train, y_train, X_test, y_test):
    """Tune the top 3 models and return the results and best overall model."""
    
    # grids for models
    rf_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 40, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    svr_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    gb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }

    xgb_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    models_to_tune = [
        (RandomForestRegressor(random_state=42), rf_grid, "Random Forest"),
        (SVR(kernel='rbf'), svr_grid, "SVR"),
        (GradientBoostingRegressor(random_state=42), gb_grid, "Gradient Boosting"),
        (XGBRegressor(random_state=42), xgb_grid, "XGBoost")
    ]

    best_models = {}
    results = []

    for estimator, param_dist, name in models_to_tune:
        print(f"\nTuning {name}...")
        start = time()
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=15,
            cv=5,
            verbose=0,
            random_state=42,
            n_jobs=-1,
            scoring='r2'
        )
        search.fit(X_train, y_train)
        duration = time() - start
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Consistent Metrics calculation
        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mae_log = mean_absolute_error(y_test, y_pred)
        
        y_test_exp = np.exp(y_test)
        y_pred_exp = np.exp(y_pred)
        mae_price = mean_absolute_error(y_test_exp, y_pred_exp)
        
        best_models[name] = best_model
        results.append({
            "Model": f"Tuned {name}",
            "R2 Score": r2,
            "Adj R2": adj_r2,
            "MAE (Log)": mae_log,
            "MAE (Price)": mae_price,
            "Best Params": str(search.best_params_),
            "Duration (s)": duration
        })
        print(f"Done in {duration:.2f}s. R2: {r2:.4f}")

    return best_models, pd.DataFrame(results)

def main():
    data_path = os.path.join(PROJ_ROOT, "data", "laptopData.csv")
    spec_path = os.path.join(PROJ_ROOT, "data", "specData.csv")
    results_dir = os.path.join(PROJ_ROOT, "results")
    model_dir = os.path.join(PROJ_ROOT, "models")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Calculate brand ratings from specData.csv
    brand_ratings = None
    if os.path.exists(spec_path):
        print("Calculating brand ratings from spec data...")
        spec_df = pd.read_csv(spec_path)
        spec_df['spec_rating'] = pd.to_numeric(spec_df['spec_rating'], errors='coerce')
        spec_df['brand_match'] = spec_df['brand'].astype(str).str.lower().str.strip()
        brand_ratings = spec_df.groupby('brand_match')['spec_rating'].mean().to_dict()

    print("Loading and preprocessing data for tuning...")
    df_raw = load_data(data_path)
    df_features = run_feature_engineering_pipeline(df_raw, brand_ratings=brand_ratings).dropna()
    X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_features)

    best_models, results_df = tune_models(X_train, y_train, X_test, y_test)

    print("\n--- Final Tuning Results ---")
    results_ordered = results_df.sort_values("R2 Score", ascending=False)
    print(results_ordered[["Model", "R2 Score", "MAE (Price)"]])

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "tuned_model_performance.csv"), index=False)

    # Save the best overall tuned model
    best_row = results_df.loc[results_df['R2 Score'].idxmax()]
    best_name = best_row['Model'].replace("Tuned ", "")
    best_model = best_models[best_name]
    
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "best_laptop_price_model_final.joblib")
    joblib.dump(best_model, model_save_path)
    
    # Save feature columns to ensure consistency during inference
    feature_cols_path = os.path.join(model_dir, "feature_columns.joblib")
    joblib.dump(X_train.columns.tolist(), feature_cols_path)
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save brand ratings
    if brand_ratings:
        brand_ratings_path = os.path.join(model_dir, "brand_ratings.joblib")
        joblib.dump(brand_ratings, brand_ratings_path)
        print(f"Brand ratings saved to {brand_ratings_path}")
    
    print(f"\nOverall winner: Tuned {best_name}")
    print(f"Final best model saved to {model_save_path}")
    print(f"Feature columns saved to {feature_cols_path}")

if __name__ == "__main__":
    main()
