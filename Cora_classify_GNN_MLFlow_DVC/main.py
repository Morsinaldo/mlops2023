import os
import hydra
import mlflow
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    # os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    # os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

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

        _ = mlflow.run(
            os.path.join(root_path, "fetch_data"),
            "main",
            parameters={
                "artifact_name": "raw_data.csv",
                "url": config["data"]["file_url"],
                "dataset": config["data"]["dataset"],
            }
        )

if __name__ == "__main__":
    process_args()