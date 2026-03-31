# Laptop Price Predictor

This project implements a machine learning pipeline to estimate laptop prices based on technical specifications. It leverages **Stacked Generalizer (Ensemble Model)** to achieve a **R² score of ~0.92**, with a Mean Absolute Error (MAE) of approximately **₹8,392** (baseline model that predicts average price for every laptop has error 29, 593).

## Highlights
1.  **CatBoost Integration:** Integrated CatBoost as a core learner for its superior handling of complex feature categorical interactions.
2.  **Stacked Generalization:** Implemented a `StackingRegressor` using XGBoost, CatBoost, Extra Trees, and SVR, blended via a `RidgeCV` meta-learner.
3.  **Hyperparameter Optimization:** Automated tuning pipeline with 20 iterations per model to find the absolute peak performance.
4.  **Brand Reputation Feature:** Dynamically calculates brand value metrics from external specification datasets.

## Feature Engineering 
The model uses a multi-stage feature engineering pipeline:
*   **Hardware Tiering:** Custom tiered mapping for CPUs (`Cpu_V2`) and GPUs (`Gpu_V2`) based on performance benchmarks.
*   **Brand Authority:** The `avg_brand_spec_rating` feature maps manufacturer quality based on historical hardware performance.
*   **Screen Specs:** Extraction of IPS panels, Touchscreen capability, and raw pixel counts from specification strings.
*   **Physical Metrics:** Normalized RAM capacity and Weight to ensure the model distinguishes between "Thin & Light" vs "Heavy Gaming" systems.

## Project Structure
*   **`src/feature_engineering.py`**: Robust data cleaning and artifact generation.
*   **`src/hyperparameter_tuning.py`**: The search engine for optimal model parameters.
*   **`src/model_training.py`**: The complete production pipeline (Baselines → Tuning → Ensemble).
*   **`src/predict.py`**: Command-line inference script for new laptop specs.
*   **`notebooks/data_cleaning.ipynb`**: Combines laptopData.csv and laptopSpecs.csv and cleans data.
*   **`notebooks/eda.ipynb`**: Detailed visual exploration and data analysis.
*   **`notebooks/feature_engineering.ipynb`**: Feature engineering and step-by-step documentation.
*   **`notebooks/model_training.ipynb`**: Model training and hyperparameter tuning.


## Quick Start
### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training the Full Ensemble
```bash
python src/hyperparameter_tuning.py
```
This script will evaluate baselines, search for the best hyperparameters, build the final ensemble, and save the winner to `models/best_laptop_price_model_final.joblib`.

### 3. Making Predictions
```bash
python src/predict.py
```

## Reproducing Metrics
*   **Train/Test Split:** 80/20 (Locked `random_state=42`).
*   **Target Scaling:** Uses `Log_Price` transformation to stabilize variance across diverse price ranges.
