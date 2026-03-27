import pandas as pd
import numpy as np
from feature_extraction import feature_extract

def cat_encode():
    df = feature_extract()

    # Feature 7: One-Hot Encoding for Categorical Variables
    categorical_cols = ['Company', 'TypeName', 'OpSys', 'CPU_core', 'GPU_brand']
    df_engineered = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Feaure 8: Dimensionality Reduction
    # Droping redundant and original text columns
    # Price drop reasoning: Created a col called log_Price that will be the new target variable
    cols_to_drop = [
         'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'Resolution_X', 'Resolution_Y', 'Price'
    ]

    df_drop = df_engineered.drop(columns=cols_to_drop)

    return df_drop
    


if __name__ == "__main__":
    df = cat_encode()
    print(df.head())

    #df.to_csv("/workspaces/Laptop-Prices/data/laptop_data_without_scale.csv", index=False)
