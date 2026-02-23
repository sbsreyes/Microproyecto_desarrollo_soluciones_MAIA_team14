import os
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("shopping_behavior_prediction")

with mlflow.start_run(run_name="RandomForest_Subscription"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("target", "Subscription")
    mlflow.log_metric("accuracy", 0.72)
    mlflow.log_metric("f1_score", 0.10)
    mlflow.log_metric("roc_auc", 0.73)
    mlflow.log_metric("rmse", 24.24)
    mlflow.log_metric("r2", -0.05)
    mlflow.log_metric("mae", 21.12)

with mlflow.start_run(run_name="LogisticRegression_Subscription"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("target", "Subscription")
    mlflow.log_metric("accuracy", 0.72)
    mlflow.log_metric("f1_score", 0.01)
    mlflow.log_metric("roc_auc", 0.71)
    mlflow.log_metric("rmse", 23.76)
    mlflow.log_metric("r2", -0.01)
    mlflow.log_metric("mae", 20.74)

print("Runs comparativos registrados")
