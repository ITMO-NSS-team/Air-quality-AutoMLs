import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import torch
import os
import random
from tqdm import tqdm
from typing import Tuple, Optional
from chronos import BaseChronosPipeline

try:
    from chronos import BaseChronosPipeline
except ImportError:
    print('Try installing Chronos via pip install '
          'git+https://github.com/amazon-science/chronos-forecasting.git')


def custom_set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


stations = ['Anchieta Centro',
            # 'Belo Horizonte',
            'Carapina',
            'Cariacica',
            'Cidade Continental',
            'Enseada do Sua',
            'Jardim Camburi',
            'Laranjeiras',
            'Mae-Ba',
            'Meaipe',
            'Ubu',
            'Vila Velha-Centro',
            'Vila Velha-IBES',
            'Vitoria-Centro']


def load_station_data(station: str, data_type: str = 'train') -> Tuple[
    np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
    features = np.load(f'../data/npy_datasets/{station}_{data_type}_features.npy')[1:]
    target = np.load(f'../data/npy_datasets/{station}_{data_type}_target.npy')[:-1]
    dates = pd.to_datetime(np.load(f'../data/npy_datasets/{station}_{data_type}_target_dates.npy'))
    return features, target, dates


def predict_with_chronos(pipeline: BaseChronosPipeline, features: np.ndarray) -> np.ndarray:
    predictions = []
    for i in range(features.shape[0]):
        context = torch.tensor(features[i], dtype=torch.float32)
        _, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        # mean shape: (batch=1, prediction_length=1)
        predictions.append(mean.item())
    return np.array(predictions)


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    mae = np.mean(np.abs(predictions - targets.ravel()))
    mse = np.sqrt(np.mean((predictions - targets.ravel()) ** 2))
    return mae, mse


def evaluate_station(pipeline: BaseChronosPipeline, station: str, chronos_model: str,
                     pbar: tqdm)-> Tuple[float, float, float, float]:
    train_features, train_target, _ = load_station_data(station, 'train')

    train_prediction = predict_with_chronos(pipeline, train_features)

    train_mae, train_mse = calculate_metrics(train_prediction, train_target)
    pbar.write(f'{station} Train MAE: {train_mae:.4f}, MSE: {train_mse:.4f}')

    test_features, test_target, target_dates = load_station_data(station, 'test')

    test_prediction = predict_with_chronos(pipeline, test_features)

    test_mae, test_mse = calculate_metrics(test_prediction, test_target)
    pbar.write(f'{station} Test MAE: {test_mae:.4f}, MSE: {test_mse:.4f}')

    plt.rcParams['figure.figsize'] = (7, 3)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.plot(target_dates[:500], test_target.ravel()[:500], label='Target', c='r', linewidth=1)
    plt.plot(target_dates[:500], test_prediction[:500], label='Prediction', c='green', linewidth=1,
             linestyle='--')
    plt.legend()
    plt.xticks(rotation=15)
    plt.ylabel('MP10, µg/m³')
    plt.title(f'Chronos LLM: {station}\nMAE={np.round(test_mae, 4)}, MSE={np.round(test_mse, 4)}')
    plt.tight_layout()
    plt.savefig(f'results/{chronos_model}/{station}.png', dpi=100)
    plt.close()

    return train_mae, train_mse, test_mae, test_mse


if __name__ == '__main__':
    custom_set_seed(42)

    train_MAE = []
    train_MSE = []
    test_MAE = []
    test_MSE = []
    AMAZON_PREFIX = 'amazon/'
    CHRONOS_MODEL = 'chronos-t5-small'

    os.makedirs(f'results/{CHRONOS_MODEL}', exist_ok=True)

    with tqdm(stations, desc="Evaluating stations") as pbar:
        for station in pbar:
            pipeline = BaseChronosPipeline.from_pretrained(
                f'{AMAZON_PREFIX}{CHRONOS_MODEL}',
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
            )
            train_mae, train_mse, test_mae, test_mse = evaluate_station(pipeline, station, CHRONOS_MODEL, pbar)
            train_MAE.append(train_mae)
            train_MSE.append(train_mse)
            test_MAE.append(test_mae)
            test_MSE.append(test_mse)

    results_df = pd.DataFrame()
    results_df['station'] = stations
    results_df['train_MAE'] = train_MAE
    results_df['train_MSE'] = train_MSE
    results_df['test_MAE'] = test_MAE
    results_df['test_MSE'] = test_MSE
    results_df.to_csv(f'results/{CHRONOS_MODEL}/metrics.csv', index=False)
