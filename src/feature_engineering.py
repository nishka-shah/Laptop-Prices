import pandas as pd
import numpy as np
import os
import joblib
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

    # Other AMD families
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
    This function processes individual rows and should be safe from leakage.
    Aggregate features like Brand Ratings should be provided in brand_ratings.
    """
    df = df.copy()
    
    # Screen features 
    if 'Resolution_X' not in df.columns or 'Resolution_Y' not in df.columns:
        res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
        df['Resolution_X'] = pd.to_numeric(res[0], errors='coerce').fillna(0).astype(int)
        df['Resolution_Y'] = pd.to_numeric(res[1], errors='coerce').fillna(0).astype(int)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and 'IPS' in x else 0)
    df['is_4K'] = df['ScreenResolution'].apply(lambda x: 1 if isinstance(x, str) and ('4K Ultra HD' in x or '3840x2160' in x) else 0)
    
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
    df['PPI'] = (((df['Resolution_X']**2) + (df['Resolution_Y']**2))**0.5 / df['Inches'].replace(0, np.nan)).fillna(0).astype('float')

    # Clean Ram and Weight
    if df['Ram'].dtype == 'O':
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False)
        df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce').fillna(8).astype(int)
    if df['Weight'].dtype == 'O':
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False)
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(2.0).astype(float)
    
    # Storage extraction
    if 'SSD' not in df.columns or 'HDD' not in df.columns:
        memory = df['Memory'].astype(str).str.replace(r'\.0', '', regex=True)
        memory = memory.str.replace('GB', '').str.replace('TB', '000')
        
        storage_info = memory.str.split("+", n=1, expand=True)
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
        df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer2HDD', 'Layer2SSD'], inplace=True, errors='ignore')

    # Hardware components
    df['CPU_Tier'] = df['Cpu'].apply(get_cpu_tier)
    df['Is_Discrete_GPU'] = df['Gpu'].apply(get_gpu_type)
    df['Has_SSD'] = (df['SSD'] > 0).astype(int)

    if 'CPU_speed' not in df.columns:
        df['CPU_speed'] = df['Cpu'].astype(str).str.extract(r'(\d+\.?\d*)GHz').astype(float).fillna(2.0)
    if 'CPU_core' not in df.columns:
        df['CPU_core'] = df['Cpu'].apply(extract_cpu_core)
    if 'GPU_brand' not in df.columns:
        df['GPU_brand'] = df['Gpu'].apply(extract_gpu_brand)

    # Performance Score
    df['Performance_Score'] = (df['CPU_Tier'] * df['CPU_speed']) + (df['Ram'] * 0.5) + (df['Is_Discrete_GPU'] * 2)
    
    # Brand Ratings (Corrected to default to global average if missing)
    avg_rating = 0.0
    if brand_ratings:
        df['avg_brand_spec_rating'] = df['Company'].astype(str).str.lower().str.strip().map(brand_ratings)
        # Use mean of provided ratings as default for unknown brands
        avg_rating = np.mean(list(brand_ratings.values())) if brand_ratings else 68.0
        df['avg_brand_spec_rating'] = df['avg_brand_spec_rating'].fillna(avg_rating)
    else:
        df['avg_brand_spec_rating'] = 68.0

    if 'Price' in df.columns:
        df['Log_Price'] = np.log(df['Price'])

    # OS Grouping
    df['OS_Category'] = df['OpSys'].apply(group_os)
    
    # Standardize casing
    categorical_cols = ['Company', 'TypeName', 'OS_Category', 'CPU_core', 'GPU_brand']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    return df

def prepare_datasets(df, target='Log_Price', test_size=0.2, random_state=42):
    """
    Splits data, handles categorical encoding and scaling with NO LEAKAGE.
    """
    # Define categorical and numerical columns
    categorical_cols = ['Company', 'TypeName', 'OS_Category', 'CPU_core', 'GPU_brand']
    
    # Split raw-ish dataframe
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Calculate Brand Ratings based ON TRAINING DATA only
    # We use Performance_Score as a proxy for brand quality if Price is not the target
    # But usually, brand reputation relates to price.
    # Note: If target is Log_Price, we use Price or Log_Price for the training set averages.
    
    # Extract brand ratings from training set prices
    if target in train_df.columns:
        brand_stats = train_df.groupby('Company')[target].mean().to_dict()
        # Update df's avg_brand_spec_rating with these training-derived values
        def map_brand(c):
            return brand_stats.get(str(c).lower().strip(), np.mean(list(brand_stats.values())))
        
        train_df['avg_brand_spec_rating'] = train_df['Company'].apply(map_brand)
        test_df['avg_brand_spec_rating'] = test_df['Company'].apply(map_brand)
    
    # Drop rows with NaNs
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # One-Hot Encoding
    # Use get_dummies on Train first to get columns, then align Test
    X_train_raw = train_df.drop(columns=[target, 'Price', 'Log_Price'], errors='ignore')
    y_train = train_df[target]
    
    X_test_raw = test_df.drop(columns=[target, 'Price', 'Log_Price'], errors='ignore')
    y_test = test_df[target]

    X_train = pd.get_dummies(X_train_raw, columns=categorical_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Drop non-numeric leftovers
    cols_to_drop = ['ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys', 'Unnamed: 0']
    X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    # Scaling
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/laptopData.csv")
    df_raw = load_data(data_path)
    df_intermediate = run_feature_engineering_pipeline(df_raw)
    X_train, X_test, y_train, y_test, scaler = prepare_datasets(df_intermediate)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
