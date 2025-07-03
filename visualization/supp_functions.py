import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load standards
standards = pd.read_csv('../data/standards.csv')
def plot_risk_zones(pollutant, period, ax=None):
    """Plot CONAMA risk zones as horizontal bands with limit line"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Get thresholds
    row = standards[(standards['pollutant'] == pollutant) &
                    (standards['period'] == period)].iloc[0]

    # Define risk zones and colors
    zones = [
        (0, row['low_max'], 'green', 'Low'),
        (row['low_max'], row['moderate_max'], 'yellow', 'Moderate'),
        (row['moderate_max'], row['high_max'], 'orange', 'High'),
        (row['high_max'], row['very_high_min']*1.2, 'red', 'Very High')
    ]

    # Plot colored zones
    for min_val, max_val, color, label in zones:
        ax.axhspan(min_val, max_val, color=color, alpha=0.2, label=label)

    # Add CONAMA limit line
    ax.axhline(row['limit'], color='black', linestyle='--', linewidth=1.5,
               label=f'CONAMA Limit ({row["limit"]} {row["unit"]})')

    # Formatting
    ax.set_ylabel(f"Concentration ({row['unit']})")
    ax.set_title(f"{pollutant} {period.upper()} Risk Zones (Brazil CONAMA)")
    ax.legend(loc='upper right')
    return ax