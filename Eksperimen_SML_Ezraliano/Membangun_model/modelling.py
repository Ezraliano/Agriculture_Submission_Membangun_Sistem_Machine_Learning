import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Set experiment dan autolog
mlflow.set_experiment("Crop Recommendation Experiment")
mlflow.autolog()

# Load dataset
df = pd.read_csv("Eksperimen_SML_Ezraliano/Membangun_model/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# ✅ Perbaikan di sini: gunakan y_train bukan X_test
model.fit(X_train, y_train)  # <-- Hanya bagian ini yang diubah

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Manual log additional metrics
mlflow.log_metric("Accuracy", acc)
mlflow.sklearn.log_model(model, "logistic_regression_model")

print(f"Model trained with Accuracy: {acc:.4f}")