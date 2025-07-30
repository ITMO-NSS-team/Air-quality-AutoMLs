import pandas as pd
from matplotlib import pyplot as plt
from visualization.supp_functions import plot_risk_zones

pollutants = ["MP10", "PTS", "NO2", "SO2", "CO", "MP2.5", "O3", "NO"]

correct_stations = [
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

dataset = pd.read_csv("../data/cleaned_dataset.csv")
dataset["Date"] = pd.to_datetime(dataset["Date"])

stations_pollutants_dict = {}
stations_target_dates = {}

for station in correct_stations:
    station_dataset = dataset[dataset["Station"] == station]
    existing_pollutants = station_dataset["Pollutant"].unique()
    stations_pollutants_dict[station] = existing_pollutants.tolist()

    station_dataset = station_dataset[station_dataset["Pollutant"] == "MP10"]
    target_dates = station_dataset["Date"][-(24 * 365) :].tolist()
    stations_target_dates[station] = (
        f'{str(target_dates[0]).split(" ")[0]} - {str(target_dates[-1]).split(" ")[0]}'
    )

    station_dataset["Value"][station_dataset["Value"] < 1] = 1
    station_dataset = station_dataset.set_index("Date").sort_index()
    station_dataset[f"24h_MP10"] = (
        station_dataset["Value"].rolling("24H", min_periods=1).mean()
    )

    plot = True
    if plot:
        # plot MP10 for each station
        """ax = plot_risk_zones('MP10', '24h')
        plt.plot(station_dataset.index, station_dataset[f'24h_MP10'])
        plt.xticks(rotation=90)
        plt.title(f'MP10 (24h mean) - {station}')
        plt.ylabel('ug/m3')
        plt.ylim(0, 240)
        plt.tight_layout()
        plt.savefig(f'mp10_visualizations/MP10_{station}.png')
        plt.show()"""

        ax = plot_risk_zones("MP10", "24h")
        plt.plot(station_dataset.index, station_dataset[f"Value"])
        plt.xticks(rotation=90)
        plt.title(f"MP10 - {station}")
        plt.ylabel("ug/m3")
        plt.ylim(0, 240)
        plt.tight_layout()
        plt.savefig(f"mp10_visualizations/raw_MP10_{station}.png")
        plt.show()

print(stations_pollutants_dict)
print(stations_target_dates)


def save_mp10_station_existance():


    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    colors = ["#ffcccc", "#ccffcc"]  # Light red and light green
    for i, station in enumerate(stations_pollutants_dict.keys()):
        for j, pollutant in enumerate(pollutants):
            if pollutant in stations_pollutants_dict[station]:
                facecolor = colors[1]  # Light green
                symbol = "✓"
            else:
                facecolor = colors[0]  # Light red
                symbol = "✗"
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=facecolor,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)
            plt.text(
                j,
                i,
                symbol,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#333333",
            )
    plt.xlim(-0.5, len(pollutants) - 0.5)
    plt.ylim(-0.5, len(stations_pollutants_dict) - 0.5)
    plt.xticks(range(len(pollutants)), pollutants, rotation=45, ha="right")
    plt.yticks(range(len(stations_pollutants_dict)), stations_pollutants_dict.keys())
    # Customize appearance
    ax.invert_yaxis()  # Put first station at top
    ax.set_axisbelow(True)
    ax.grid(which="major", color="white", linestyle="-", linewidth=2)
    ax.tick_params(axis="both", which="both", length=0)  # Hide tick marks
    # Add title and adjust layout
    plt.title("Pollutant Monitoring by Station", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig("pollutant_per_station.png", dpi=400)
    plt.show()


def save_test_dates():


    def convert_date_format(date_str):
        start, end = date_str.split(" - ")
        return f"{'/'.join(start.split('-'))} - {'/'.join(end.split('-'))}"

    stations = list(stations_target_dates.keys())
    dates = [convert_date_format(stations_target_dates[s]) for s in stations]
    fig, ax = plt.subplots(figsize=(4.5, 6))  # Increased height
    ax.axis("off")
    table_data = [["Station", "Target Date Range"]] + [
        [s, d] for s, d in zip(stations, dates)
    ]
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="left",
        colWidths=[0.35, 0.65],
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("none")
        cell.set_edgecolor("#cccccc")
        cell.set_height(0.1)  # Additional height control
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_edgecolor("#999999")
    plt.tight_layout()
    plt.savefig("station_target_dates.png", dpi=400, bbox_inches="tight")
    plt.show()


# save_mp10_station_existance()
# save_test_dates()
