import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fedotllm import run_assistant
from sklearn.metrics import mean_absolute_error, mean_squared_error

DESCRIPTION_PROMPT = """
This competition focuses on air quality prediction using data from Brazilian air monitoring stations.
Develop accurate model to predict MP10 concentrations.
The goal is to create reliable predictions.
Models are evaluated using MAE metric
"""

stations = [
    "Anchieta Centro",
    #'Belo Horizonte',
    "Carapina",
    "Cariacica",
    "Cidade Continental",
    "Enseada do Sua",
    "Jardim Camburi",
    "Laranjeiras",
    "Mae-Ba",
    "Meaipe",
    "Ubu",
    "Vila Velha-Centro",
    "Vila Velha-IBES",
    "Vitoria-Centro",
]

global_train_MAE = []
global_train_MSE = []
global_test_MAE = []
global_test_MSE = []


def create_visualization(
    test_target: np.ndarray,
    test_prediction: np.ndarray,
    station: str,
    automl: str,
    mae: float,
    mse: float,
):
    test_target = np.ravel(test_target)
    test_prediction = np.ravel(test_prediction)
    target_dates = pd.to_datetime(
        np.load(f"./data/npy_datasets/{station}_test_target_dates.npy")
    )

    plt.rcParams["figure.figsize"] = (7, 3)
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.plot(target_dates[:500], test_target[:500], label="Target", c="r", linewidth=1)
    plt.plot(
        target_dates[:500],
        test_prediction[:500],
        label="Prediction",
        c="green",
        linewidth=1,
        linestyle="--",
    )
    plt.legend()
    plt.xticks(rotation=15)
    plt.ylabel("MP10, µg/m³")
    plt.title(f"FEDOT.ASSISTANT: {station}\nMAE={np.round(mae, 4)}, MSE={np.round(mse, 4)}")
    plt.tight_layout()
    plt.savefig(f"./baselines/results/fedotllm/regression/{automl}/{station}.png", dpi=100)
    plt.close()


def save_metrics(
    name: str,
    automl: str,
    station: list,
    train_MAE: list,
    train_MSE: list,
    test_MAE: list,
    test_MSE: list,
):
    results = pd.DataFrame(
        {
            "Station": station,
            "Train_MAE": train_MAE,
            "Train_MSE": train_MSE,
            "Test_MAE": test_MAE,
            "Test_MSE": test_MSE,
        }
    )
    results.to_csv(f"./baselines/results/fedotllm/regression/{automl}/{name}.csv", index=False)


def run_fedotllm(train: pd.DataFrame, test: pd.DataFrame, station: str, automl: str):
    with tempfile.TemporaryDirectory(prefix="_datasets", dir=Path.cwd()) as tmp_dir:
        temp_dir = Path(tmp_dir).resolve()
        test_features = test.drop(columns=["Target_MP10"])
        test_target = test["Target_MP10"].to_numpy()

        train.to_csv(temp_dir / "train.csv", header=True, index=False)
        test_features.to_csv(temp_dir / "test.csv", header=True, index=False)
        pd.DataFrame({"Target_MP10": [123.4567, 1.2345]}).to_csv(
            temp_dir / "sample_submission.csv", header=True, index=False
        )

        description = temp_dir / "description.txt"
        description.write_text(DESCRIPTION_PROMPT)
        output_filename = temp_dir / "output.csv"

        task, assistant = run_assistant(
            task_path=temp_dir,
            presets="best_quality",
            config_overrides=[f"automl.enabled={automl}", 
                            "time_limit=1800"],
            output_filename=output_filename,
        )

        test_prediction = pd.read_csv(output_filename)
        test_prediction = test_prediction["Target_MP10"].to_numpy()

        test_mae = mean_absolute_error(test_target, test_prediction)
        test_mse = mean_squared_error(test_target, test_prediction)
        print(f"Test MAE: {test_mae}")
        print(f"Test MSE: {test_mse}")

        create_visualization(test_target, test_prediction, station, automl, test_mae, test_mse)

        # Training prediction
        task.test_data = task.train_data.drop(columns=["Target_MP10"])
        train_target = task.train_data["Target_MP10"].to_numpy()

        train_prediction = assistant.predict(task).to_numpy()

        train_mae = mean_absolute_error(train_target, train_prediction)
        train_mse = mean_squared_error(train_target, train_prediction)
        print(f"Train MAE: {train_mae}")
        print(f"Train MSE: {train_mse}")
        
        return test_mae, test_mse, train_mae, train_mse


def process_station(station: str, automl: str):
    train = pd.read_csv(f"./data/csv_datasets/{station}_train.csv")
    test = pd.read_csv(f"./data/csv_datasets/{station}_test.csv")

    test_MAE, test_MSE, train_MAE, train_MSE = run_fedotllm(train, test, station, automl)

    global_train_MAE.append(train_MAE)
    global_train_MSE.append(train_MSE)
    global_test_MAE.append(test_MAE)
    global_test_MSE.append(test_MSE)

    save_metrics(
        f"{station}_metrics",
        automl,
        [station],
        [train_MAE],
        [train_MSE],
        [test_MAE],
        [test_MSE],
    )


if __name__ == "__main__":
    automls = ["fedot", "autogluon"]
    for automl in automls:
        for station in stations:
            process_station(station, automl)
        save_metrics(
            "metrics",
            automl,
            stations,
            global_train_MAE,
            global_train_MSE,
            global_test_MAE,
            global_test_MSE,
        )
