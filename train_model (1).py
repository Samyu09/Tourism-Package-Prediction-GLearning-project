
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump

# Load train data
train_df = pd.read_csv("train.csv")

# Separate features and target
X = train_df.drop(columns=["ProdTaken"])
y = train_df["ProdTaken"]

# Identify categorical and numerical columns
cat_cols = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
model.fit(X_processed, y)

# Save model and preprocessor
dump(model, "model.joblib")
dump(preprocessor, "preprocessor.joblib")

print("Model and preprocessor saved successfully.")
