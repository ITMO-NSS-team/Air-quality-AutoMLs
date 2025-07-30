import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

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


def moving_average_forecast(X):


    predictions = np.mean(X, axis=1)
    return predictions


train_MAE = []
train_MSE = []
test_MAE = []
test_MSE = []


for station in stations:
    train_features = np.load(f"../data/npy_datasets/{station}_train_features.npy")[1:]
    train_target = np.load(f"../data/npy_datasets/{station}_train_target.npy")[:-1]

    train_prediction = moving_average_forecast(train_features)

    train_mae = np.mean(abs(train_prediction - train_target))
    train_mse = np.sqrt((np.mean((train_prediction - train_target) ** 2)))

    print(f"Train MAE: {train_mae}")
    print(f"Train MSE: {train_mse}")
    train_MAE.append(train_mae)
    train_MSE.append(train_mse)

    test_features = np.load(f"../data/npy_datasets/{station}_test_features.npy")[1:]
    test_target = np.load(f"../data/npy_datasets/{station}_test_target.npy")[:-1]

    test_prediction = moving_average_forecast(test_features)

    test_mae = np.mean(abs(test_prediction - test_target))
    test_mse = np.sqrt((np.mean((test_prediction - test_target) ** 2)))

    print(f"Test MAE: {test_mae}")
    print(f"Test MSE: {test_mse}")
    test_MAE.append(test_mae)
    test_MSE.append(test_mse)

    test_target = np.ravel(test_target)
    test_prediction = np.ravel(test_prediction)
    target_dates = pd.to_datetime(
        np.load(f"../data/npy_datasets/{station}_test_target_dates.npy")
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
        f"Moving average: {station}\nMAE={np.round(test_mae, 4)}, MSE={np.round(test_mse, 4)}"
    )
    plt.tight_layout()
    plt.savefig(f"results/moving_average/{station}_after_fix.png", dpi=100)
    # plt.show()
    plt.close()


results_df = pd.DataFrame()
results_df["station"] = stations
results_df["train_MAE"] = train_MAE
results_df["train_MSE"] = train_MSE
results_df["test_MAE"] = test_MAE
results_df["test_MSE"] = test_MSE
results_df.to_csv("results/moving_average/metrics_after_fix.csv", index=False)
