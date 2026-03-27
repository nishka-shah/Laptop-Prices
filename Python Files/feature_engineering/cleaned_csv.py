import pandas as pd
import numpy as np



def load_data():
    df = pd.read_csv("/workspaces/Laptop-Prices/data/cleaned_laptop_data.csv")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    

