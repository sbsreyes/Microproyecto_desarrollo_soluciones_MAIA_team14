import mlflow
import random
import os

mlflow.set_experiment("microproyecto_demo")

with mlflow.start_run():
    seed = 42
    random.seed(seed)

    accuracy = random.uniform(0.7, 0.95)

    mlflow.log_param("seed", seed)
    mlflow.log_param("modelo", "baseline_random")
    mlflow.log_metric("accuracy", accuracy)

    os.makedirs("reports", exist_ok=True)
    with open("reports/demo_output.txt", "w") as f:
        f.write(f"Accuracy obtenida: {accuracy}")

    mlflow.log_artifact("reports/demo_output.txt")

print("Run registrado en MLflow correctamente.")
