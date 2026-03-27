from sklearn.preprocessing import StandardScaler
from train_test_split import train_test
import pandas as pd

X_train, X_test, y_train, y_test = train_test()

# Select only numerical features for scaling
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Standardization (mean=0, std=1) ensures all features contribute equally
scaler = StandardScaler()

# Fit on train, transform both train and test
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(X_train[numeric_cols])

#X_train[numeric_cols].to_csv("/workspaces/Laptop-Prices/data/laptop_data_with_scale.csv", index=False)


