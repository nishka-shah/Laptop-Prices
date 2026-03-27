import pandas as pd
import numpy as np
from cleaned_csv import load_data


def feature_extract():
    """
    Performs feature engineering to extract meaningful signals from raw laptop data.
    Focuses on reducing dimensionality and handling skewed distributions.
    """

    df = load_data()


    # Feature 1: Binary Encoding for Display Tech
    # Specific panel types like IPS and Touch capability are premium 
    # markers that significantly drive price variations
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

    # Feature 2: Data Type Normalization
    # Ensures 'Inches' is treated as a continuous numerical variable for calculations
    df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')

    # Feature 3: Pixels per inches (PPI)
    # ResolutionX and ResoultionY are highly correlated so PPI combines resolution and screen size
    # Captures higher quality and denser display
    df['PPI'] = (((df['Resolution_X']**2) + (df['Resolution_Y']**2))**0.5 / df['Inches']).astype('float')

    # Feature 4: Target Transformation(Log Scale)
    # Through log transformation, we have a compression of scale
    # To normalize the distribution and minimize the impact of high-end 'outlier' prices.
    df['Log_Price'] = np.log(df['Price'])

    # Feature 5: Rare Label Encoding
    # To group the less common companies (<10 appearences)
    company_counts = df['Company'].value_counts()
    rare_companies = company_counts[company_counts < 10].index
    df['Company'] = df['Company'].apply(lambda x: 'Other' if x in rare_companies else x)

    # Feature 6: Performance Index
    # Looking at the combination of Ram and CPU as a fast CPU with low RAM (or vice versa) creates a bottleneck
    # Highlights high end laptops
    df['Performance_Index'] = df['Ram'] * df['CPU_speed']

    return df

if __name__ == "__main__":
    df = feature_extract()    
    print(df.head())
    