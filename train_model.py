import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset load
df = pd.read_csv("dataset/diabetes.csv")

print(df.head())

# Input and Output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Save Model
pickle.dump(
    model,
    open("model/diabetes_model.pkl", "wb")
)

print("Model Saved Successfully")