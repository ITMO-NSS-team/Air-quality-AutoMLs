import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATS

RESULTS_PATH = (
    Path(__file__).parent.parent / "baselines" / "results" / "autonbeats_forecast"
)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

stations = [
    "Anchieta Centro",
    # 'Belo Horizonte',
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

if __name__ == "__main__":
    raw_df = pd.read_csv(Path(__file__).parent.parent / "data" / "cleaned_dataset.csv")
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    all_results = []
    evaluated_stations = set(map(lambda x: x.split(".")[0], os.listdir(RESULTS_PATH)))
    for station in stations:
        if station in evaluated_stations:
            continue
        df = raw_df[raw_df["Station"] == station].copy()
        df = df[df["Pollutant"] == "MP10"]
        df = df.sort_values("Date")
        df.loc[df["Value"] < 1, "Value"] = 1
        df = df.set_index("Date", drop=False).sort_index()
        df["24h_MP10"] = df["Value"].rolling("24h", min_periods=1).mean()
        df = df.reset_index(drop=True)
        target_dates = df["Date"][-(24 * 365) :].tolist()
        test_df = df[df["Date"].isin(target_dates)].copy()
        train_df = df[~df["Date"].isin(target_dates)].copy()

        train_df = train_df.rename(columns={"Date": "ds", "24h_MP10": "y"})[["ds", "y"]]
        train_df["unique_id"] = station
        test_df = test_df.rename(columns={"Date": "ds", "24h_MP10": "y"})[["ds", "y"]]
        test_df["unique_id"] = station

        horizon = 48
        n_rolls = int(np.ceil(len(test_df) / horizon))
        preds = []
        true_vals = []
        ds_vals = []

        model = AutoNBEATS(h=horizon, backend="optuna")
        nf = NeuralForecast(models=[model], freq="h", local_scaler_type="standard")
        nf.fit(df=train_df)

        test_y = test_df["y"].values
        test_ds = test_df["ds"].values
        for i in range(n_rolls):
            start_idx = i * horizon
            end_idx = min((i + 1) * horizon, len(test_df))
            context_df = pd.concat([train_df, test_df.iloc[:start_idx]])
            y_hat_df = nf.predict(df=context_df)
            pred_chunk = y_hat_df[y_hat_df["unique_id"] == station][
                "AutoNBEATS"
            ].values[: end_idx - start_idx]
            preds.append(pred_chunk)
            true_vals.append(test_y[start_idx:end_idx])
            ds_vals.append(test_ds[start_idx:end_idx])

        preds = np.concatenate(preds)
        true_vals = np.concatenate(true_vals)
        ds_vals = np.concatenate(ds_vals)
        mae = np.mean(np.abs(true_vals - preds))
        mse = np.sqrt(np.mean((true_vals - preds) ** 2))
        all_results.append({"station": station, "test_MAE": mae, "test_MSE": mse})

        plt.rcParams["figure.figsize"] = (7, 3)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
        plt.plot(ds_vals[:500], true_vals[:500], label="Target", c="r", linewidth=1)
        plt.plot(
            ds_vals[:500],
            preds[:500],
            label="Prediction",
            c="green",
            linewidth=1,
            linestyle="--",
        )
        plt.legend()
        plt.xticks(rotation=15)
        plt.ylabel("MP10, µg/m³")
        plt.title(
            f"AutoNBEATS (rolling): {station}\nMAE={np.round(mae, 4)}, MSE={np.round(mse, 4)}"
        )
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / f"{station}.png", dpi=100)
        plt.close()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH / "metrics.csv", index=False)
