from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

stations_names = [
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


def load_data(station, window_size=48):
    """
    No valid docstring found.
    """

    dataset = pd.read_csv("data/cleaned_dataset.csv")
    dataset["Date"] = pd.to_datetime(dataset["Date"])

    station_dataset = dataset[dataset["Station"] == station]

    station_dataset = station_dataset[station_dataset["Pollutant"] == "MP10"]
    station_dataset = station_dataset.sort_values("Date")

    target_dates = station_dataset["Date"][-(24 * 365) :].tolist()

    station_dataset["Value"][station_dataset["Value"] < 1] = 1
    station_dataset = station_dataset.set_index("Date", drop=False).sort_index()
    station_dataset[f"24h_MP10"] = (
        station_dataset["Value"].rolling("24H", min_periods=1).mean()
    )

    test_dataset = station_dataset[station_dataset["Date"].isin(target_dates)]
    train_dataset = station_dataset[~station_dataset["Date"].isin(target_dates)]

    plt.rcParams["figure.figsize"] = (6, 2)
    plt.plot(train_dataset["Date"], train_dataset["24h_MP10"], label="Train")
    plt.plot(test_dataset["Date"], test_dataset["24h_MP10"], label="Test")
    # plt.title(f'Train/Test time series - {station}')
    plt.title(f"{station}")
    plt.legend(loc="upper left")
    plt.ylabel("MP10, µg/m³")
    plt.tight_layout()
    plt.show()

    # FORM TRAIN DATASET
    """train_features = []
    train_target = []
    train_target_dates = []
    for date in train_dataset['Date'].tolist():
        print(f'{station} - Train: Process {date}')
        prehistory_dates = pd.date_range(date-timedelta(hours=window_size), date, freq='H', inclusive='left')
        prehistory_vals = []
        for d in prehistory_dates:
            try:
                val = train_dataset['24h_MP10'][train_dataset['Date'] == d].values[0]
                prehistory_vals.append(val)
            except Exception:
                break
        if len(prehistory_vals) == window_size:
            train_features.append(prehistory_vals)
            train_target.append([train_dataset['24h_MP10'][train_dataset['Date'] == date].values[0]])
            train_target_dates.append(str(date))

    train_features = np.array(train_features)
    np.save(f'data/npy_datasets/{station}_train_features.npy', train_features)
    train_target = np.array(train_target)
    np.save(f'data/npy_datasets/{station}_train_target.npy', train_target)
    train_target_dates = np.array(train_target_dates)
    np.save(f'data/npy_datasets/{station}_train_target_dates.npy', train_target_dates)


    # FORM TEST DATASET
    test_features = []
    test_target = []
    test_target_dates = []
    for date in test_dataset['Date'].tolist():
        print(f'{station} - Test: Process {date}')
        prehistory_dates = pd.date_range(date - timedelta(hours=window_size), date, freq='H', inclusive='left')
        prehistory_vals = []
        for d in prehistory_dates:
            try:
                val = test_dataset['24h_MP10'][test_dataset['Date'] == d].values[0]
                prehistory_vals.append(val)
            except Exception:
                break
        if len(prehistory_vals) == window_size:
            test_features.append(prehistory_vals)
            test_target.append([test_dataset['24h_MP10'][test_dataset['Date'] == date].values[0]])
            test_target_dates.append(str(date))

    test_features = np.array(test_features)
    np.save(f'data/npy_datasets/{station}_test_features.npy', test_features)
    test_target = np.array(test_target)
    np.save(f'data/npy_datasets/{station}_test_target.npy', test_target)
    test_target_dates = np.array(test_target_dates)
    np.save(f'data/npy_datasets/{station}_test_target_dates.npy', test_target_dates)"""


for station in stations_names:
    load_data(station)
