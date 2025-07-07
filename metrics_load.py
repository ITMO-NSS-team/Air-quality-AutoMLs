import pandas as pd

models = ['linear_regression', 'moving_average', 'naive_forecast']
models_path = 'baselines/results'

metric = 'MSE'
final_table = pd.DataFrame()

for model in models:
    file_name = f'{models_path}/{model}/metrics.csv'
    df = pd.read_csv(file_name)
    final_table['station'] = df['station']
    final_table[f'{model} train'] = df[f'train_{metric}']
    final_table[f'{model} test'] = df[f'test_{metric}']

final_table.to_csv(f'baselines_{metric}.csv', index=False)

