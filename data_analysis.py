import os
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt

from visualization.supp_functions import plot_risk_zones

path = "C:/Users/Julia/Documents/NSS_lab/документы/2025 Air quality paper/raw_data"

dfs = []
for file in os.listdir(path):
    df = pd.read_csv(f"{path}/{file}")
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.columns = [
    "Date",
    "Time",
    "Station",
    "Code",
    "Pollutant",
    "Value",
    "Unit",
    "Type",
]
combined_df["Date"] = pd.to_datetime(combined_df["Date"] + " " + combined_df["Time"])

for c in ["Station", "Code", "Pollutant", "Unit", "Type"]:
    print(f"{c} - {combined_df[c].unique().tolist()}")

combined_df["Station"] = combined_df["Station"].replace(
    "Vitória Centro", "Vitoria-Centro"
)
combined_df["Station"] = combined_df["Station"].replace(
    "Vitoria Centro", "Vitoria-Centro"
)
combined_df["Station"] = combined_df["Station"].replace(
    "Cariacica Vila Capixaba", "Cariacica"
)
combined_df["Station"] = combined_df["Station"].replace(
    "Enseada do Suá", "Enseada do Sua"
)
combined_df["Station"] = combined_df["Station"].replace(
    "Vila Velha - Centro", "Vila Velha-Centro"
)
combined_df["Station"] = combined_df["Station"].replace(
    "Vila Velha - Ibes", "Vila Velha-IBES"
)

# combined_df.to_csv('data/cleaned_dataset.csv', index=False)

stations = combined_df["Station"].unique()
print(stations)
pollutants = combined_df["Pollutant"].unique()


def save_ts():
    """
    Function saves plots with multidimensional time series for each station

    """

    for station in stations:
        print(station)
        station_df = combined_df[combined_df["Station"] == station]
        for pollutant in pollutants:
            pollutant_df = station_df[station_df["Pollutant"] == pollutant]
            if len(pollutant_df) != 0:
                plt.plot(pollutant_df["Date"], pollutant_df["Value"], label=pollutant)
        plt.title(station)
        plt.legend()
        plt.ylabel("μg/m³")
        plt.yscale("log")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(f"data/ts_visualizations/{station}.png", dpi=100)
        plt.close()


station_name = "Laranjeiras"
pollutant = "MP10"

single_station = combined_df[combined_df["Station"] == station_name]
single_station = single_station[single_station["Pollutant"] == pollutant]

single_station = single_station.set_index("Date").sort_index()
single_station[f"24h_{pollutant}"] = (
    single_station["Value"].rolling("24H", min_periods=1).mean()
)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d - %H:%M"))
plt.rcParams["figure.figsize"] = (10, 6)
ax = plot_risk_zones(pollutant, "24h")
plt.plot(single_station.index[-2000:-1000], single_station["Value"][-2000:-1000])
plt.xticks(rotation=90)
plt.title(f"{pollutant} - {station_name}")
plt.ylabel("ug/m3")
plt.tight_layout()
plt.show()

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d - %H:%M"))
plt.rcParams["figure.figsize"] = (10, 6)
ax = plot_risk_zones(pollutant, "24h")
plt.plot(
    single_station.index[-2000:-1000], single_station[f"24h_{pollutant}"][-2000:-1000]
)
plt.xticks(rotation=90)
plt.title(f"{pollutant} - {station_name}")
plt.ylabel("ug/m3")
plt.tight_layout()
plt.show()
