# Laptop Price Predictor

This project uses machine learning to estimate the price of a laptop based on its hardware specifications (CPU, RAM, GPU, screen quality, etc.). It reaches about 91-92% accuracy (R²) using a Gradient Boosting model.

### Folder Structure
- **`data/`**: contains the raw `laptopData.csv` and the supplemental `specData.csv` for brand reputation ratings.
- **`notebooks/`**: exploration and prototyping. I’ve documented the feature engineering and model training logic here. 
- **`src/`**: the core logic. `feature_engineering.py` handles the data cleaning, and `predict.py` is what you use for actual inferences.
- **`models/`**: serialized artifacts like the trained model, the scaler, and feature mappings.
- **`results/`**: logs of how different models performed.

### Quick Start
1. **Prepare the environment**: make sure you have the dependencies installed (requirements.txt).
2. **Train the model**: run the hyperparameter tuning script to generate the artifacts.
   ```bash
   python3 src/hyperparameter_tuning.py
   ```
3. **Run a prediction**: use the prediction script to see how the model handles new data.
   ```bash
   python3 src/predict.py
   ```

### Dependencies
You'll need Python 3 and the following libraries:
- `pandas` & `numpy` (for data manipulation)
- `scikit-learn` (the ML framework)
- `joblib` (for loading/saving models)
- `matplotlib` & `seaborn` (if you want to run the notebooks)

### Reproducing the Results
If you want to get the exact same numbers I did:
- The training/test split is always **80/20**.
- The `random_state` is locked at **42** across all scripts.
- We pull in brand reputation metrics from `specData.csv` to give the model a hint about build quality/brand value.
- The target variable is `Log_Price` to handle the variance in high-end vs budget laptops.
