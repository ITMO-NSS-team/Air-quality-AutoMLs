import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

stations = [#'Anchieta Centro',
            'Belo Horizonte',
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

for station in stations:
    train_features = np.load(f'../data/npy_datasets/{station}_train_features.npy')
    train_target = np.load(f'../data/npy_datasets/{station}_train_target.npy')

    reg = LinearRegression().fit(train_features, train_target)
    train_prediction = reg.predict(train_features)

    train_mae = np.mean(abs(train_prediction - train_target))
    train_mse = np.mean(np.sqrt((train_prediction - train_target) ** 2))

    print(f'Train MAE: {train_mae}')
    print(f'Train MSE: {train_mse}')

    test_features = np.load(f'../data/npy_datasets/{station}_test_features.npy')
    test_target = np.load(f'../data/npy_datasets/{station}_test_target.npy')

    test_prediction = reg.predict(test_features)

    test_mae = np.mean(abs(test_prediction - test_target))
    test_mse = np.mean(np.sqrt((test_prediction - test_target) ** 2))

    print(f'Train MAE: {test_mae}')
    print(f'Train MSE: {test_mse}')

    test_target = np.ravel(test_target)
    test_prediction = np.ravel(test_prediction)
    target_dates = pd.to_datetime(np.load(f'../data/npy_datasets/{station}_test_target_dates.npy'))

    plt.rcParams['figure.figsize'] = (10, 4)
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.plot(target_dates[:50], test_target[:50], label='Target', c='r', linewidth=1)
    plt.plot(target_dates[:50], test_prediction[:50], label='Prediction', c='green', linewidth=1,
             linestyle='--')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title(f'Linear regression: {station}\nMAE={np.round(test_mae, 4)}, MSE={np.round(test_mse, 4)}')
    plt.tight_layout()
    plt.show()


