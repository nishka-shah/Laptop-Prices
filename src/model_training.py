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

# Import all modeling requirements
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# Visual settings
sns.set_theme(style="whitegrid")

def calculate_price_metrics(y_true_log, y_pred_log):
    """Calculate evaluation metrics on the back-transformed price scale."""
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2 Score": r2_score(y_true_log, y_pred_log),
        "MAE (Price)": mean_absolute_error(y_true, y_pred),
        "MSE (Price)": mse,
        "RMSE (Price)": np.sqrt(mse),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

def tune_models(X_train, y_train, X_test, y_test):
    """Tune the top models using RandomizedSearchCV."""
    rf_grid = {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, None], 'max_features': ['sqrt', 'log2', None]}
    gb_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5, 6], 'subsample': [0.8, 0.9, 1.0]}
    xgb_grid = {'n_estimators': [100, 300, 500, 1000], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [4, 6, 8, 10], 'subsample': [0.7, 0.8, 1.0]}
    cat_grid = {'iterations': [100, 500, 1000], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8, 10], 'l2_leaf_reg': [1, 3, 5, 7]}
    et_grid = {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    svr_grid = {'C': [0.1, 1, 10, 100, 500], 'epsilon': [0.01, 0.1, 0.2], 'gamma': ['scale', 'auto', 0.1, 0.01]}

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
        print(f"Tuning {name}...")
        start = time()
        search = RandomizedSearchCV(
            estimator=estimator, param_distributions=param_dist,
            n_iter=20, cv=5, verbose=0, random_state=42, n_jobs=-1, scoring='r2'
        )
        search.fit(X_train, y_train)
        duration = time() - start
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = calculate_price_metrics(y_test, y_pred)
        best_models[name] = best_model
        
        res = {"Model": f"Tuned {name}", "Duration (s)": duration}
        res.update(metrics)
        results.append(res)
        
        print(f"Done. R2: {metrics['R2 Score']:.4f}")

    # Build Final Tuned Stacked Generalizer (Ensemble)
    print("\nBuilding Final Tuned Stacked Generalizer...")
    final_tuned_stack = StackingRegressor(
        estimators=[
            ('xgb', best_models["XGBoost"]),
            ('cat', best_models["CatBoost"]),
            ('et', best_models["Extra Trees"]),
            ('svr', best_models["SVR"])
        ],
        final_estimator=RidgeCV(), cv=5, n_jobs=-1
    )
    
    final_tuned_stack.fit(X_train, y_train)
    y_pred_stack = final_tuned_stack.predict(X_test)
    metrics_stack = calculate_price_metrics(y_test, y_pred_stack)
    
    best_models["Stacked Ensemble"] = final_tuned_stack
    res_stack = {"Model": "Tuned Stacked Generalizer (Ensemble)", "Duration (s)": 0}
    res_stack.update(metrics_stack)
    results.append(res_stack)

    return best_models, pd.DataFrame(results)

def evaluate_simple(models, X_train, X_test, y_train, y_test):
    """Basic evaluation for initial comparison."""
    results = []
    for name, model in models.items():
        start_time = time()
        model.fit(X_train, y_train)
        duration = time() - start_time
        y_pred = model.predict(X_test)
        
        metrics = calculate_price_metrics(y_test, y_pred)
        res = {"Model": name, "Time (s)": duration}
        res.update(metrics)
        results.append(res)
    return pd.DataFrame(results)

def main():
    data_path = os.path.join(PROJ_ROOT, "data", "laptopData.csv")
    report_dir = os.path.join(PROJ_ROOT, "reports")
    results_dir = os.path.join(PROJ_ROOT, "results")
    model_dir = os.path.join(PROJ_ROOT, "models")

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Loading and preprocessing data (NO LEAKAGE split)...")
    df_raw = load_data(data_path)
    
    # Run spec-based feature engineering (Internal row-level features)
    df_intermediate = run_feature_engineering_pipeline(df_raw)
    
    # Split and Encode (Handles Brand Ratings via Training Mean Price)
    X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_intermediate)

    # Baseline Training
    print("\nPhase 1: Baseline Comparison")
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "CatBoost": CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
    }

    baseline_df = evaluate_simple(baseline_models, X_train, X_test, y_train, y_test)
    print(baseline_df.sort_values(by="R2 Score", ascending=False)[["Model", "R2 Score", "MAE (Price)"]])

    # Hyperparameter Tuning
    print("\nPhase 2: Hyperparameter Tuning")
    best_models, final_df = tune_models(X_train, y_train, X_test, y_test)
    
    final_ordered = final_df.sort_values(by="R2 Score", ascending=False)
    print(final_ordered[["Model", "R2 Score", "MAE (Price)", "RMSE (Price)", "MAPE"]])
    final_ordered.to_csv(os.path.join(results_dir, "final_tuned_performance.csv"), index=False)

    # Save Winner
    best_overall_name = final_ordered.iloc[0]["Model"]
    model_key = "Stacked Ensemble" if "Ensemble" in best_overall_name else best_overall_name.replace("Tuned ", "")
    best_model = best_models[model_key]

    joblib.dump(best_model, os.path.join(model_dir, "best_laptop_price_model_final.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "feature_columns.joblib"))
    
    # Recalculate brand ratings from training set for saving into production artifacts
    brand_stats = pd.concat([X_train_orig_company_info(df_intermediate, X_train.index), y_train], axis=1).groupby('Company')['Log_Price'].mean().to_dict()
    joblib.dump(brand_stats, os.path.join(model_dir, "brand_ratings.joblib"))

    print(f"\nWINNER: {best_overall_name} saved to models/")

def X_train_orig_company_info(df_orig, train_indices):
    """Helper to retrieve original company column for final artifact saving."""
    return df_orig.loc[train_indices, 'Company']

if __name__ == "__main__":
    main()
