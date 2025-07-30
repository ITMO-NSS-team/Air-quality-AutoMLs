import os

import numpy as np
import pandas as pd

stations_names = [
    "Anchieta Centro",
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

# Directories for input and output data
input_dir = "data/npy_datasets"
output_dir = "data/csv_datasets"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Process each station
for station in stations_names:
    print(f"Processing station: {station}")
    # Process both train and test sets
    for set_type in ["train", "test"]:
        try:
            # Form file paths
            features_path = os.path.join(
                input_dir, f"{station}_{set_type}_features.npy"
            )
            target_path = os.path.join(input_dir, f"{station}_{set_type}_target.npy")
            # Check if all three files exist
            if not all(os.path.exists(p) for p in [features_path, target_path]):
                print(
                    f"  - Skipping '{set_type}' set, as one or more files were not found."
                )
                continue

            # Load NumPy arrays
            features = np.load(features_path)[1:]
            target = np.load(target_path)[:-1]
            # allow_pickle=True is needed as dates might be saved as objects
            # Determine the window size from the data
            window_size = features.shape[1]

            # Create column names for features (t-N, ..., t-1, t-0)
            feature_cols = [f"t-{i}" for i in range(window_size - 1, -1, -1)]

            # Create a DataFrame for features
            df_features = pd.DataFrame(features, columns=feature_cols)

            # Create a DataFrame for targets and dates
            df_target = pd.DataFrame(
                {"Target_MP10": target.flatten()}  # .flatten() to convert to 1D array
            )

            # Concatenate the two DataFrames column-wise
            final_df = pd.concat([df_target, df_features], axis=1)

            # Form the path for saving the CSV file
            output_path = os.path.join(output_dir, f"{station}_{set_type}.csv")

            # Save the result to CSV
            final_df.to_csv(output_path, index=False)

            print(f"  - Saved '{set_type}' data to: {output_path}")

        except Exception as e:
            print(
                f"  - An error occurred while processing '{set_type}' for station {station}: {e}"
            )

print("\nConversion complete.")
