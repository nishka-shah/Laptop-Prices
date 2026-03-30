import nbformat as nbf
import os

def sync_notebook_to_script():
    path = '/Users/nishkashah/Documents/laptop-price/notebooks/model_training.ipynb'
    with open(path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # New synchronized content
    new_source = (
        "import os\n"
        "import pandas as pd\n"
        "from src.feature_engineering import load_data, run_feature_engineering_pipeline, prepare_datasets\n"
        "\n"
        "# 1. Define paths\n"
        "data_path = \"../data/laptopData.csv\"\n"
        "spec_path = \"../data/specData.csv\"\n"
        "\n"
        "# 2. Calculate brand ratings from specData.csv (for improved accuracy)\n"
        "brand_ratings = None\n"
        "if os.path.exists(spec_path):\n"
        "    print(\"Calculating brand ratings from spec data...\")\n"
        "    spec_df = pd.read_csv(spec_path)\n"
        "    spec_df['spec_rating'] = pd.to_numeric(spec_df['spec_rating'], errors='coerce')\n"
        "    spec_df['brand_match'] = spec_df['brand'].astype(str).str.lower().str.strip()\n"
        "    brand_ratings = spec_df.groupby('brand_match')['spec_rating'].mean().to_dict()\n"
        "\n"
        "# 3. Run the robust feature engineering pipeline with raw data\n"
        "df_raw = load_data(data_path)\n"
        "df_features = run_feature_engineering_pipeline(df_raw, brand_ratings=brand_ratings)\n"
        "df_features = df_features.dropna()\n"
        "\n"
        "# 4. Split and Scale\n"
        "X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_features)\n"
        "\n"
        "print(f\"Data loaded from: {data_path}\")\n"
        "print(f\"Total features after encoding: {X_train.shape[1]}\")\n"
        "print(f\"Training set: {X_train.shape}\")\n"
        "print(f\"Testing set: {X_test.shape}\")"
    )

    updated_count = 0
    # Loop and replace content of ANY cell that looks like an old-style loading cell
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'cleaned_laptop_data.csv' in cell.source and 'run_feature_engineering_pipeline' in cell.source:
                cell.source = new_source
                updated_count += 1
            # Also update the one that might already have laptopData.csv but missing brand_ratings
            elif 'laptopData.csv' in cell.source and 'run_feature_engineering_pipeline' in cell.source and 'brand_ratings' not in cell.source:
                cell.source = new_source
                updated_count += 1

    with open(path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook synchronized. Updated {updated_count} cells.")

if __name__ == "__main__":
    sync_notebook_to_script()
