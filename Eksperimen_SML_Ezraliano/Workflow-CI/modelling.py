import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Set tracking URI DagsHub (opsional)
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Ezraliano/Agriculture_Submission_Membangun_Sistem_Machine_Learning.mlflow "
os.environ["MLFLOW_TRACKING_USERNAME"] = "Ezraliano"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "eb75b25073f06d07c77a57aee65dd85f94545a0c"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Load dataset
df_path = os.path.abspath("Eksperimen_SML_Ezraliano/dataset_raw/Crop_recommendation.csv")
df = pd.read_csv(df_path)

# Pisahkan fitur dan target
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Start run MLflow
with mlflow.start_run():
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_params(model.get_params())
    mlflow.sklearn.log_model(model, "logistic_regression_model")

print(f"Model trained with Accuracy: {acc:.4f}")