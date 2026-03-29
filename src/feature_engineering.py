import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads CSV data from path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    return pd.read_csv(filepath)

def extract_cpu_core(cpu):
    import re
    cpu = str(cpu).lower()

    # Intel
    if "i3" in cpu:
        return "i3"
    elif "i5" in cpu:
        return "i5"
    elif "i7" in cpu:
        return "i7"
    elif "i9" in cpu:
        return "i9"

    # AMD Ryzen
    elif "ryzen 3" in cpu or "ryzen3" in cpu:
        return "ryzen3"
    elif "ryzen 5" in cpu or "ryzen5" in cpu:
        return "ryzen5"
    elif "ryzen 7" in cpu or "ryzen7" in cpu:
        return "ryzen7"
    elif "ryzen 9" in cpu or "ryzen9" in cpu:
        return "ryzen9"

    # Other AMD families seen in the laptop dataset
    elif "a4" in cpu:
        return "amd_a4"
    elif "a6" in cpu:
        return "amd_a6"
    elif "a8" in cpu:
        return "amd_a8"
    elif "a9" in cpu:
        return "amd_a9"
    elif "a10" in cpu:
        return "amd_a10"
    elif "a12" in cpu:
        return "amd_a12"
    elif "e-series" in cpu or re.search(r'\be2\b', cpu):
        return "amd_e"
    elif "fx" in cpu:
        return "amd_fx"
    elif "athlon" in cpu:
        return "athlon"

    else:
        return "other"

def extract_gpu_brand(gpu):
    gpu = str(gpu).lower()
    if "nvidia" in gpu:
        return "nvidia"
    elif "amd" in gpu or "radeon" in gpu:
        return "amd"
    elif "intel" in gpu:
        return "intel"
    else:
        return "other"

def get_cpu_tier(cpu):
    """Categorizes CPU into performance tiers (0-5)."""
    cpu = str(cpu).lower()
    if 'i9' in cpu or 'xeon' in cpu or 'ryzen 9' in cpu: return 5
    if 'i7' in cpu or 'ryzen 7' in cpu: return 4
    if 'i5' in cpu or 'ryzen 5' in cpu: return 3
    if 'i3' in cpu or 'ryzen 3' in cpu: return 2
    if 'celeron' in cpu or 'pentium' in cpu or 'atom' in cpu: return 0
    return 1 

def get_gpu_type(gpu):
    """Classifies GPU as Discrete (1) or Integrated (0)."""
    gpu = str(gpu).lower()
    discrete_keywords = ['gtx', 'rtx', 'quadro', 'radeon pro', 'radeon rx', 'firepro', 'm1000', 'm1200', 'm2000', 'm2200']
    for word in discrete_keywords:
        if word in gpu: return 1 
    return 0

def group_os(os_name):
    """Groups operating systems into major categories."""
    os_name = str(os_name).lower()
    if 'windows' in os_name: return 'Windows'
    if 'mac' in os_name or 'macos' in os_name: return 'Mac'
    if 'linux' in os_name: return 'Linux'
    if 'no os' in os_name: return 'No OS'
    return 'Other'

def run_feature_engineering_pipeline(df, brand_ratings=None):
    """
    Modular feature engineering pipeline.
    Expects a dataframe from 'laptopData.csv' (raw or cleaned).
    """
    df = df.copy()
    
    # Screen features 
    # Extract resolution if missing (needed for inference on raw specs)
    if 'Resolution_X' not in df.columns or 'Resolution_Y' not in df.columns:
        res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
        df['Resolution_X'] = pd.to_numeric(res[0], errors='coerce').fillna(0).astype(int)
        df['Resolution_Y'] = pd.to_numeric(res[1], errors='coerce').fillna(0).astype(int)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and 'IPS' in x else 0)
    df['is_4K'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and ('4K Ultra HD' in x or '3840x2160' in x) else 0)
    
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
    df['PPI'] = (((df['Resolution_X']**2) + (df['Resolution_Y']**2))**0.5 / df['Inches'].replace(0, np.nan)).fillna(0).astype('float')

    # Clean Ram and Weight if they are strings
    if df['Ram'].dtype == 'O':
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False)
        df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce').fillna(8).astype(int)
    if df['Weight'].dtype == 'O':
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False)
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(2.0).astype(float)
    
    # Storage extraction (SSD, HDD) from Memory string if columns are missing
    if 'SSD' not in df.columns or 'HDD' not in df.columns:
        df['Memory'] = df['Memory'].astype(str).str.replace(r'\.0', '', regex=True)
        df["Memory"] = df["Memory"].str.replace('GB', '')
        df["Memory"] = df["Memory"].str.replace('TB', '000')
        
        storage_info = df["Memory"].str.split("+", n=1, expand=True)
        df['first'] = storage_info[0].str.strip()
        df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
        
        df['first'] = df['first'].str.extract(r'(\d+)').fillna(0).astype(int)
        
        if storage_info.shape[1] > 1:
            df['second'] = storage_info[1].fillna("0")
        else:
            df["second"] = "0"
            
        df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
        df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
        
        df['second'] = df['second'].str.extract(r'(\d+)').fillna(0).astype(int)
        
        df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
        df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
        df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer2HDD', 'Layer2SSD'], inplace=True)

    # Hardware components
    df['CPU_Tier'] = df['Cpu'].apply(get_cpu_tier)
    df['Is_Discrete_GPU'] = df['Gpu'].apply(get_gpu_type)
    
    # Logic for Has_SSD (Binary indicator)
    df['Has_SSD'] = (df['SSD'] > 0).astype(int)

    # 3. Hardware details extraction if missing
    if 'CPU_speed' not in df.columns:
        df['CPU_speed'] = df['Cpu'].astype(str).str.extract(r'(\d+\.?\d*)GHz').astype(float).fillna(2.0)
    if 'CPU_core' not in df.columns:
        df['CPU_core'] = df['Cpu'].apply(extract_cpu_core)
    if 'GPU_brand' not in df.columns:
        df['GPU_brand'] = df['Gpu'].apply(extract_gpu_brand)

    # Performance Score and Brand Ratings
    df['Performance_Score'] = (df['CPU_Tier'] * df['CPU_speed']) + (df['Ram'] * 0.5) + (df['Is_Discrete_GPU'] * 2)
    
    if brand_ratings:
        df['avg_brand_spec_rating'] = df['Company'].astype(str).str.lower().str.strip().map(brand_ratings)
        # Fill missing brand ratings with global average (e.g. ~68)
        df['avg_brand_spec_rating'] = df['avg_brand_spec_rating'].fillna(68.0)
    else:
        # If no ratings provided, use placeholder
        df['avg_brand_spec_rating'] = 68.0

    if 'Price' in df.columns:
        df['Log_Price'] = np.log(df['Price'])

    # Handle Categorical grouping
    # Rare Label Encoding for Company
    company_counts = df['Company'].value_counts()
    rare_companies = company_counts[company_counts < 10].index
    df['Company'] = df['Company'].apply(lambda x: 'Other' if x in rare_companies else x)
    
    # OS Grouping
    df['OS_Category'] = df['OpSys'].apply(group_os)
    
    # One-Hot Encoding
    categorical_cols = ['Company', 'TypeName', 'OS_Category', 'CPU_core', 'GPU_brand']
    
    # Standardize casing for categorical columns to prevent casing errors during inference
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    df_engineered = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Drop redundant columns
    # We keep SSD and HDD as they are original features
    cols_to_drop = [
        'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Price', 
        'Resolution_X', 'Resolution_Y', 'Pixel_Count', 'Company_match', 'Unnamed: 0'
    ]
    df_final = df_engineered.drop(columns=[c for c in cols_to_drop if c in df_engineered.columns])
    
    # Final cleanup: Drop rows with NaNs (ex. from Inches coercion)
    df_final = df_final.dropna()
    
    return df_final

def prepare_datasets(df, target='Log_Price', test_size=0.2, random_state=42):
    """Splits data and scales numerical features."""
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Identify numerical columns for scaling and ensure float dtype
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train.loc[:, numeric_cols] = X_train[numeric_cols].astype(float)
    X_test.loc[:, numeric_cols] = X_test[numeric_cols].astype(float)

    scaler = StandardScaler()
    
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_laptop_data.csv")
    print(f"Loading data from {data_path}...")
    
    try:
        df_raw = load_data(data_path)
        df_features = run_feature_engineering_pipeline(df_raw)
        X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_features)
        
        print("Pipeline Execution Successful!")
        print(f"Engineered Features Shape: {df_features.shape}")
        print(f"Training Set Shape: {X_train.shape}")
        print(f"Target variable statistics:\n{y_train.describe()}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
