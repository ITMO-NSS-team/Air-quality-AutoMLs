from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedotllm import PredictionAssistant, PredictionTask, load_config
from sklearn.metrics import mean_absolute_error, mean_squared_error

DESCRIPTION_PROMPT = """
This competition focuses on air quality forecasting using time series data from Brazilian air monitoring stations.
Develop accurate time series forecasting models to predict MP10 concentrations.
The goal is to create reliable hourly forecast.
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

train_MAE = []
train_MSE = []
test_MAE = []
test_MSE = []

# FEDOT Pipeline
node_lagged_1 = PrimaryNode("lagged")
node_final = SecondaryNode("ridge", nodes_from=[node_lagged_1])
pipeline = Pipeline(node_final)

for station in stations:
    train_features = np.load(f"./data/npy_datasets/{station}_train_features.npy")[1:]
    train_target = np.load(f"./data/npy_datasets/{station}_train_target.npy")[:-1]

    HORIZON = 1
    WINDOW = len(train_features[0])
    node_lagged_1.parameters = {"window_size": WINDOW - 2}

    train = train_features[0, :].tolist()
    for i in range(1, len(train_features)):
        train.append(train_features[i, -1])
    train = np.array(train)
    train = pd.DataFrame({"target_MP10": train.flatten()})

    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)

    train.to_csv(temp_dir / "train.csv", header=True, index=False)

    description = Path(temp_dir) / "description.txt"
    description.write_text(DESCRIPTION_PROMPT)
    output_filename = Path(temp_dir) / "output.csv"

    # Load config
    config = load_config(
        presets="medium_quality",
        overrides=["automl.enabled=fedot"],
    )
    config.automl.fedot.predictor_init_kwargs["n_jobs"] = -1
    config.time_limit = 120

    # Create assistant
    assistant = PredictionAssistant(config)

    # Create and preprocess task
    prediction_task = PredictionTask.from_path(temp_dir.resolve())
    prediction_task = assistant.preprocess_task(prediction_task)

    # Create pipeline with correct parameter order
    node_lagged_1 = PrimaryNode("lagged")
    node_lagged_1.parameters = {"window_size": WINDOW - 2}

    node_final = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    pipeline = Pipeline(node_final)

    # Setup FEDOT task
    fedot_task = Task(
        TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=HORIZON)
    )

    # Prepare training data correctly
    train_data_input = prediction_task.train_data["target_MP10"].to_numpy().squeeze()
    train_input = InputData(
        idx=np.arange(0, len(train_data_input)),
        features=train_data_input,
        target=train_data_input,
        task=fedot_task,
        data_type=DataTypesEnum.ts,
    )

    # Fit pipeline
    pipeline.fit(
        train_input, n_jobs=config.automl.fedot.predictor_init_kwargs["n_jobs"]
    )

    # Training prediction loop
    start_forecast = 0
    end_train_forecast = train_features.shape[0]
    forecast_train_idx = np.arange(start_forecast, end_train_forecast)

    train_prediction = []
    for i in train_features:
        start_forecast = len(i)
        end_forecast = start_forecast + HORIZON
        forecast_idx = np.arange(start_forecast, end_forecast)
        predict_train_input = InputData(
            idx=forecast_idx,
            features=i,
            target=i,
            task=fedot_task,
            data_type=DataTypesEnum.ts,
        )
        predicted_output = pipeline.predict(predict_train_input)
        train_prediction += np.ravel(np.array(predicted_output.predict)).tolist()

    train_mae = mean_absolute_error(train_target, train_prediction)
    train_mse = mean_squared_error(train_target, train_prediction)

    print(f"Train MAE: {train_mae}")
    print(f"Train MSE: {train_mse}")

    train_MAE.append(train_mae)
    train_MSE.append(train_mse)

    # Testing prediction loop
    test_features = np.load(f"./data/npy_datasets/{station}_test_features.npy")[1:]
    test_target = np.load(f"./data/npy_datasets/{station}_test_target.npy")[:-1]
    start_forecast = 0
    end_test_forecast = test_features.shape[0]
    forecast_test_idx = np.arange(start_forecast, end_test_forecast)
    test_prediction = []
    for i in test_features:
        start_forecast = len(i)
        end_forecast = start_forecast + HORIZON
        forecast_idx = np.arange(start_forecast, end_forecast)
        predict_test_input = InputData(
            idx=forecast_idx,
            features=i,
            target=i,
            task=fedot_task,
            data_type=DataTypesEnum.ts,
        )
        predicted_output = pipeline.predict(predict_test_input)
        test_prediction += np.ravel(np.array(predicted_output.predict)).tolist()

    test_mae = mean_absolute_error(test_target, test_prediction)
    test_mse = mean_squared_error(test_target, test_prediction)

    print(f"Test MAE: {test_mae}")
    print(f"Test MSE: {test_mse}")

    test_MAE.append(test_mae)
    test_MSE.append(test_mse)

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
    plt.title(
        f"FEDOTLLM: {station}\nMAE={np.round(test_mae, 4)}, MSE={np.round(test_mse, 4)}"
    )
    plt.tight_layout()
    plt.savefig(f"./baselines/results/fedotllm/univariate/{station}.png", dpi=100)
    # plt.show()
    plt.close()

    results_df = pd.DataFrame()
    results_df["station"] = station
    results_df["train_MAE"] = train_MAE
    results_df["train_MSE"] = train_MSE
    results_df["test_MAE"] = test_MAE
    results_df["test_MSE"] = test_MSE
    results_df.to_csv(
        f"./baselines/results/fedotllm/univariate/{station}_metrics.csv", index=False
    )


results_df = pd.DataFrame()
results_df["station"] = stations
results_df["train_MAE"] = train_MAE
results_df["train_MSE"] = train_MSE
results_df["test_MAE"] = test_MAE
results_df["test_MSE"] = test_MSE
results_df.to_csv("./baselines/results/fedotllm/univariate/metrics.csv", index=False)
