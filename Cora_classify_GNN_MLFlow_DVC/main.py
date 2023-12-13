import os
import hydra
import mlflow
from omegaconf import DictConfig

artifact_folder = "artifacts"
figures_folder = "figures"

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step
    if "fetch_data" in steps_to_execute:
        with mlflow.start_run():
            _ = mlflow.run(
                os.path.join(root_path, "fetch_data"),
                "main",
                parameters={
                    "artifact_folder": artifact_folder,
                }
            )
            # Adiciona os artefatos ao controle do DVC
            os.system(f"dvc add ./{artifact_folder}/citations.csv")
            os.system(f"dvc add ./{artifact_folder}/papers.csv")
            os.system(f"dvc push")

        # _ = mlflow.run(
        #     os.path.join(root_path, "fetch_data"),
        #     "main",
        #     parameters={
        #         "artifact_folder": artifact_folder,
        #     }
        # )
    
    # EDA step
    if "eda" in steps_to_execute:

        with mlflow.start_run():
            _ = mlflow.run(
                os.path.join(root_path, "eda"),
                "main",
                parameters={
                    "figures_folder": figures_folder,
                    "artifact_folder": artifact_folder,
                }
            )
            # Adiciona os artefatos ao controle do DVC
            os.system(f"dvc add ./{figures_folder}/citations.png")
            os.system(f"dvc add ./{figures_folder}/papers.png")
            os.system(f"dvc push")

        # _ = mlflow.run(
        #     os.path.join(root_path, "eda"),
        #     "main",
        #     parameters={
        #         "figures_folder": figures_folder,
        #         "artifact_folder": artifact_folder,
        #     }
        # )


    # Preprocessing step
    if "preprocessing" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "preprocessing"),
            "main",
            parameters={
                "figures_folder": figures_folder,
                "artifact_folder": artifact_folder,
            }
        )

    # Data segregation step
    if "data_segregation" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "data_segregation"),
            "main",
            parameters={
                "artifact_folder": artifact_folder,
            }
        )

    # Training step
    if "train" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "train"),
            "main",
            parameters={
                "artifact_folder": artifact_folder,
                "figures_folder": figures_folder,
            }
        )

if __name__ == "__main__":
    process_args()