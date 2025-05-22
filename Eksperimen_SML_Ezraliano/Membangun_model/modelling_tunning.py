import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# 🔁 Set tracking URI untuk DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Ezraliano/Agriculture_Submission_Membangun_Sistem_Machine_Learning.mlflow "
os.environ["MLFLOW_TRACKING_USERNAME"] = "Ezraliano"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "eb75b25073f06d07c77a57aee65dd85f94545a0c"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Load dataset
df = pd.read_csv("Eksperimen_SML_Ezraliano/Membangun_model/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000, 1500]
}

with mlflow.start_run():
    grid = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log params and metrics manually
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("F1 Score", f1)

    # Simpan model ke DagsHub
    mlflow.sklearn.log_model(best_model, "best_logistic_regression_model")

    print(f"Best Parameters: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")