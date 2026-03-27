from categorical_encoding import cat_encode
from sklearn.model_selection import train_test_split

def train_test():
    """
    Splits the engineered dataset into training and testing sets.
    Target variable is the Log_Price.
    """
    df = cat_encode()

    # Separate features (X) from the target (y)
    X = df.drop(columns=['Log_Price'])
    y = df['Log_Price']

    # Splitting: 80% Training & 20% Testing
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test()
