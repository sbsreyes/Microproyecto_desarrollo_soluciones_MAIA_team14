import mlflow
import random

mlflow.set_experiment("xray-entrega2")

with mlflow.start_run():
    lr = 0.0005
    epochs = 15
    accuracy = 0.83

    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("accuracy", accuracy)

    print("Experimento registrado correctamente")
