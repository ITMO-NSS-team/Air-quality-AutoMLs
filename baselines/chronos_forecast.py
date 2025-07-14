import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import torch
import os
try:
    from chronos import BaseChronosPipeline
except ImportError:
    print('Try installing Chronos via pip install '
          'git+https://github.com/amazon-science/chronos-forecasting.git')


stations = ['Anchieta Centro',
            #'Belo Horizonte',
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

if __name__ == '__main__':
    test_MAE = []
    test_MSE = []
    AMAZON_PREFIX = 'amazon/'
    CHRONOS_MODEL = 'chronos-t5-small'

    pipeline = BaseChronosPipeline.from_pretrained(
        f'{AMAZON_PREFIX}{CHRONOS_MODEL}',
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
    )
    os.makedirs(f'results/{CHRONOS_MODEL}', exist_ok=True)

    for station in stations:
        test_features = np.load(f'../data/npy_datasets/{station}_test_features.npy')[1:]
        test_target = np.load(f'../data/npy_datasets/{station}_test_target.npy')[:-1]
        target_dates = pd.to_datetime(np.load(f'../data/npy_datasets/{station}_test_target_dates.npy'))

        test_prediction = []
        for i in range(test_features.shape[0]):
            context = torch.tensor(test_features[i], dtype=torch.float32)
            _, mean = pipeline.predict_quantiles(
                context=context,
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            # mean shape: (batch=1, prediction_length=1)
            test_prediction.append(mean.item())
        test_prediction = np.array(test_prediction)

        test_mae = np.mean(np.abs(test_prediction - test_target.ravel()))
        test_mse = np.sqrt(np.mean((test_prediction - test_target.ravel()) ** 2))

        print(f'{station} Test MAE: {test_mae}')
        print(f'{station} Test MSE: {test_mse}')
        test_MAE.append(test_mae)
        test_MSE.append(test_mse)


        plt.rcParams['figure.figsize'] = (7, 3)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
        plt.plot(target_dates[:500], test_target.ravel()[:500], label='Target', c='r', linewidth=1)
        plt.plot(target_dates[:500], test_prediction[:500], label='Prediction', c='green', linewidth=1, linestyle='--')
        plt.legend()
        plt.xticks(rotation=15)
        plt.ylabel('MP10, µg/m³')
        plt.title(f'Chronos LLM: {station}\nMAE={np.round(test_mae, 4)}, MSE={np.round(test_mse, 4)}')
        plt.tight_layout()
        plt.savefig(f'results/{CHRONOS_MODEL}/{station}.png', dpi=100)
        plt.close()

    results_df = pd.DataFrame()
    results_df['station'] = stations
    results_df['test_MAE'] = test_MAE
    results_df['test_MSE'] = test_MSE
    results_df.to_csv(f'results/{CHRONOS_MODEL}/metrics.csv', index=False)