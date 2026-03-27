import seaborn as sns
import matplotlib.pyplot as plt
from train_test_split import train_test

X_train, X_test, y_train, y_test = train_test()
plt.figure(figsize=(14,10))

# Correlation Matric
# Helps identify multi-collinearity
sns.heatmap(
    X_train.corr(numeric_only=True),
    cmap="coolwarm",
    center = 0
)

plt.title("Corelation Matrix")

plt.savefig("correlation_matrix.png", bbox_inches='tight')
plt.show()