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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

def tune_models(X_train, y_train, X_test, y_test):
    """Tune the top models and return the best estimators."""
    
    # Parameter Grids
    rf_grid = {
        'n_estimators': [100, 300, 500], 
        'max_depth': [10, 20, None], 
        'max_features': ['sqrt', 'log2', None]
    }
    gb_grid = {
        'n_estimators': [100, 300, 500], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0]
    }
    xgb_grid = {
        'n_estimators': [100, 300, 500, 1000], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'max_depth': [4, 6, 8, 10], 
        'subsample': [0.7, 0.8, 1.0]
    }
    cat_grid = {
        'iterations': [100, 500, 1000], 
        'learning_rate': [0.01, 0.05, 0.1], 
        'depth': [4, 6, 8, 10], 
        'l2_leaf_reg': [1, 3, 5, 7]
    }
    et_grid = {
        'n_estimators': [100, 300, 500], 
        'max_depth': [10, 20, None], 
        'min_samples_split': [2, 5, 10]
    }
    svr_grid = {
        'C': [0.1, 1, 10, 100, 500], 
        'epsilon': [0.01, 0.1, 0.2], 
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    models_to_tune = [
        (RandomForestRegressor(random_state=42), rf_grid, "Random Forest"),
        (GradientBoostingRegressor(random_state=42), gb_grid, "Gradient Boosting"),
        (XGBRegressor(random_state=42), xgb_grid, "XGBoost"),
        (CatBoostRegressor(verbose=0, random_state=42), cat_grid, "CatBoost"),
        (ExtraTreesRegressor(random_state=42), et_grid, "Extra Trees"),
        (SVR(kernel='rbf'), svr_grid, "SVR")
    ]

    best_models = {}
    results = []

    for estimator, param_dist, name in models_to_tune:
        print(f"\nTuning {name}...")
        start = time()
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=20, 
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
        
        r2 = r2_score(y_test, y_pred)
        n, p = X_test.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mae_price = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        
        best_models[name] = best_model
        results.append({
            "Model": f"Tuned {name}",
            "R2 Score": r2,
            "Adj R2": adj_r2,
            "MAE (Price)": mae_price,
            "Best Params": str(search.best_params_),
            "Duration (s)": duration
        })
        print(f"Done in {duration:.2f}s. R2: {r2:.4f}")

    # Build Final Stacked Generalizer (Ensemble)
    print("\nBuilding Final Tuned Stacked Generalizer...")
    final_tuned_stack = StackingRegressor(
        estimators=[
            ('xgb', best_models["XGBoost"]),
            ('cat', best_models["CatBoost"]),
            ('et', best_models["Extra Trees"]),
            ('svr', best_models["SVR"])
        ],
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    start_stack = time()
    final_tuned_stack.fit(X_train, y_train)
    stack_duration = time() - start_stack
    
    y_pred_stack = final_tuned_stack.predict(X_test)
    r2_stack = r2_score(y_test, y_pred_stack)
    n, p = X_test.shape
    adj_r2_stack = 1 - (1 - r2_stack) * (n - 1) / (n - p - 1)
    mae_price_stack = mean_absolute_error(np.exp(y_test), np.exp(y_pred_stack))
    
    best_models["Stacked Ensemble"] = final_tuned_stack
    results.append({
        "Model": "Tuned Stacked Generalizer (Ensemble)",
        "R2 Score": r2_stack,
        "Adj R2": adj_r2_stack,
        "MAE (Price)": mae_price_stack,
        "Best Params": "Ensemble of tuned models",
        "Duration (s)": stack_duration
    })
    print(f"Ensemble trained in {stack_duration:.2f}s. R2: {r2_stack:.4f}")

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

    print("\n--- Final Leaderboard ---")
    results_ordered = results_df.sort_values("R2 Score", ascending=False)
    print(results_ordered[["Model", "R2 Score", "MAE (Price)"]])

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "tuned_model_performance.csv"), index=False)

    # Identify and save the winner
    best_row = results_ordered.iloc[0]
    best_overall_name = best_row['Model']
    
    # Map friendly name back to best_models key
    model_key = None
    if "Ensemble" in best_overall_name: model_key = "Stacked Ensemble"
    else: model_key = best_overall_name.replace("Tuned ", "")
    
    best_model = best_models[model_key]
    
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "best_laptop_price_model_final.joblib")
    joblib.dump(best_model, model_save_path)
    
    # Save standard artifacts
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "feature_columns.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    if brand_ratings:
        joblib.dump(brand_ratings, os.path.join(model_dir, "brand_ratings.joblib"))
    
    print(f"\n WINNER: {best_overall_name}")
    print(f"Final Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
