
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, accuracy_score

# Load test data
test_df = pd.read_csv("test.csv")
X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]

# Load saved model and preprocessor
model = load("model.joblib")
preprocessor = load("preprocessor.joblib")

# Preprocess test data
X_test_processed = preprocessor.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_processed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
