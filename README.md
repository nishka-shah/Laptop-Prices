# Advanced Laptop Price Predictor (Ensemble Optimization)

This project implements a high-performance machine learning pipeline to estimate laptop prices based on technical specifications. It leverages an advanced **Stacked Generalizer (Ensemble Model)** to achieve a professional-grade **R² score of ~0.92**, with a Mean Absolute Error (MAE) of approximately **₹8,350**.

## 🚀 Recent Updates
1.  **CatBoost Integration:** Integrated CatBoost as a core learner for its superior handling of complex feature categorical interactions.
2.  **Stacked Generalization:** Implemented a `StackingRegressor` using XGBoost, CatBoost, Random Forest, Extra Trees, and SVR, blended via a `RidgeCV` meta-learner.
3.  **Hyperparameter Optimization:** Automated tuning pipeline with 20 iterations per model to find the absolute peak performance.
4.  **Brand Reputation Feature:** Dynamically calculates brand value metrics from external specification datasets.

## 📊 Performance Leaderboard (Final Results)
| Model | R² Score | MAE (Price) |
| :--- | :--- | :--- |
| **🏆 Tuned Stacked Generalizer** | **0.9179** | **₹8,348** |
| Tuned CatBoost | 0.9147 | ₹8,748 |
| Tuned XGBoost | 0.9121 | ₹8,499 |
| Tuned Gradient Boosting | 0.9107 | ₹8,713 |
| Tuned Random Forest | 0.9015 | ₹8,921 |

## 🛠️ Feature Engineering Suite
The model uses a multi-stage feature engineering pipeline:
*   **Hardware Tiering:** Custom tiered mapping for CPUs (`Cpu_V2`) and GPUs (`Gpu_V2`) based on performance benchmarks.
*   **Brand Authority:** The `avg_brand_spec_rating` feature maps manufacturer quality based on historical hardware performance.
*   **Screen Specs:** Extraction of IPS panels, Touchscreen capability, and raw pixel counts from specification strings.
*   **Physical Metrics:** Normalized RAM capacity and Weight to ensure the model distinguishes between "Thin & Light" vs "Heavy Gaming" systems.

### Top Predictors (Best Features)
Based on Feature Importance analysis, the following features have the highest impact on price:
1.  **RAM:** The single strongest predictor of cost.
2.  **CPU Tier (V2):** Performance tiering of the processor.
3.  **Weight:** Strongly correlates with build materials and form factor.
4.  **Brand Spec Rating:** Custom brand quality metric.
5.  **Gpu_V2:** High-end vs. integrated graphics tiering.

## 📂 Project Structure
*   **`src/feature_engineering.py`**: Robust data cleaning and artifact generation.
*   **`src/hyperparameter_tuning.py`**: The search engine for optimal model parameters.
*   **`src/model_training.py`**: The complete production pipeline (Baselines → Tuning → Ensemble).
*   **`src/predict.py`**: Command-line inference script for new laptop specs.
*   **`notebooks/model_training.ipynb`**: Detailed visual exploration and step-by-step documentation.

## 🚦 Quick Start
### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training the Full Ensemble
```bash
python src/model_training.py
```
This script will evaluate baselines, search for the best hyperparameters, build the final ensemble, and save the winner to `models/best_laptop_price_model_final.joblib`.

### 3. Making Predictions
```bash
python src/predict.py
```

## 📉 Reproducing Metrics
*   **Train/Test Split:** 80/20 (Locked `random_state=42`).
*   **Target Scaling:** Uses `Log_Price` transformation to stabilize variance across diverse price ranges.
*   **Ensemble BLEND:** Uses five diverse learners (Boosters, Bagging, and Geometric models) to minimize residual error.
