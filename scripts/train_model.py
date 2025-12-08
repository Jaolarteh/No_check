import yaml
import mlflow
from src.models.train import train_all_models

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_experiment(cfg["training"]["experiment_name"])

    with mlflow.start_run():
        results = train_all_models(
            cfg["data"]["path"],
            cfg["data"]["target_col"],
            cfg["training"]["test_size"],
            cfg["training"]["random_state"]
        )

        for model_name, metrics in results.items():
            for k, v in metrics.items():
                if k == "confusion_matrix":
                    mlflow.log_dict({model_name: v}, f"{model_name}_cm.json")
                else:
                    mlflow.log_metric(f"{model_name}_{k}", v)

if __name__ == "__main__":
    main()
