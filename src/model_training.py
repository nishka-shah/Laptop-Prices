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

# Import all modeling requirements directly since we are consolidating
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Visual settings
sns.set_theme(style="whitegrid")

def tune_models(X_train, y_train, X_test, y_test):
    """Tune the top models using RandomizedSearchCV and return the best estimators."""
    
    # Parameter Grids matching the high-performance notebook
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
        print(f"\nTuning {name}...")
        start = time()
        search = RandomizedSearchCV(
            estimator=estimator, param_distributions=param_dist,
            n_iter=20, cv=5, verbose=0, random_state=42, n_jobs=-1, scoring='r2'
        )
        search.fit(X_train, y_train)
        duration = time() - start
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae_price = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        
        best_models[name] = best_model
        results.append({"Model": f"Tuned {name}", "R2 Score": r2, "MAE (Price)": mae_price, "Duration (s)": duration})
        print(f"Done in {duration:.2f}s. R2: {r2:.4f}")
        print(f"Optimal Parameters: {search.best_params_}")

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
    
    start_stack = time()
    final_tuned_stack.fit(X_train, y_train)
    stack_duration = time() - start_stack
    
    y_pred_stack = final_tuned_stack.predict(X_test)
    r2_stack = r2_score(y_test, y_pred_stack)
    mae_price_stack = mean_absolute_error(np.exp(y_test), np.exp(y_pred_stack))
    
    best_models["Stacked Ensemble"] = final_tuned_stack
    results.append({"Model": "Tuned Stacked Generalizer (Ensemble)", "R2 Score": r2_stack, "MAE (Price)": mae_price_stack, "Duration (s)": stack_duration})
    print(f"Ensemble trained in {stack_duration:.2f}s. R2: {r2_stack:.4f}")

    return best_models, pd.DataFrame(results)

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
        results.append({"Model": name, "R2 Score": r2, "MAE (Price)": mae_price, "Time (s)": duration})
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

    # Brand Ratings calculation
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

    # Baseline Training
    print("\n Phase 1: Baseline Comparison")
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
            estimators=[('xgb', XGBRegressor(n_estimators=100, random_state=42)), ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)), ('et', ExtraTreesRegressor(n_estimators=100, random_state=42)), ('svr', SVR(kernel='rbf', C=10))],
            final_estimator=RidgeCV(), cv=5, n_jobs=-1
        )
    }

    baseline_df = evaluate_simple(baseline_models, X_train, X_test, y_train, y_test)
    print(baseline_df.sort_values(by="R2 Score", ascending=False)[["Model", "R2 Score", "MAE (Price)"]])

    # Hyperparameter Tuning & Tuned Ensemble
    print("\n Phase 2: Hyperparameter Tuning Finalists")
    best_models, final_df = tune_models(X_train, y_train, X_test, y_test)
    
    print("\n Phase 3: Final Tuned Leaderboard")
    final_ordered = final_df.sort_values(by="R2 Score", ascending=False)
    print(final_ordered[["Model", "R2 Score", "MAE (Price)"]])
    
    final_ordered.to_csv(os.path.join(results_dir, "final_tuned_performance.csv"), index=False)

    # Save Production Winner
    best_overall_name = final_ordered.iloc[0]["Model"]
    model_key = "Stacked Ensemble" if "Ensemble" in best_overall_name else best_overall_name.replace("Tuned ", "")
    best_model = best_models[model_key]

    model_save_path = os.path.join(model_dir, "best_laptop_price_model_final.joblib")
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "feature_columns.joblib"))
    if brand_ratings:
        joblib.dump(brand_ratings, os.path.join(model_dir, "brand_ratings.joblib"))
    
    print(f"\n WINNER: {best_overall_name}")
    print(f"R² Score: {final_ordered.iloc[0]['R2 Score']:.4f}")
    print(f"Final Model saved to: {model_save_path}")

    # Final Plot
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkviolet')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Production Model Fit: {best_overall_name}")
    plt.savefig(os.path.join(report_dir, "production_model_fit.png"))
    plt.close()
    print(f"Performance plot saved to {report_dir}")

    # Feature Importance Analysis (Absolute Winner)
    print("\n Phase 4: Feature Importance Analysis")
    best_overall_name = final_ordered.iloc[0]["Model"]
    model_key = "Stacked Ensemble" if "Ensemble" in best_overall_name else best_overall_name.replace("Tuned ", "")
    best_model_instance = best_models[model_key]

    print(f"Analyzing importance for the leaderboard winner: {best_overall_name}")

    if hasattr(best_model_instance, 'feature_importances_'):
        print(f"Using built-in feature importances for {best_overall_name}...")
        importances = best_model_instance.feature_importances_
        method = "Built-in"
    else:
        print(f"Calculating Permutation Importance for {best_overall_name} (this may take a moment)...")
        r = permutation_importance(best_model_instance, X_test, y_test,
                                  n_repeats=10,
                                  random_state=42,
                                  n_jobs=-1)
        importances = r.importances_mean
        method = "Permutation"

    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='mako', hue=feat_imp.index, legend=False)
    plt.title(f"Top 20 Feature Importances ({best_overall_name}) - {method} Method")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "feature_importances.png"))
    plt.close()
    print(f"Feature importance plot ({method}) saved to {report_dir}")

if __name__ == "__main__":
    main()
