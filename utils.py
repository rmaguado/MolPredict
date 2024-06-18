import os
import pandas as pd


def save(output_folder, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_paths = [
        os.path.join(output_folder, f'{item}.csv') for item in ['train', 'test', 'val']
    ]

    train = train.drop_duplicates(ignore_index=True)
    test = test.drop_duplicates(ignore_index=True)
    val = val.drop_duplicates(ignore_index=True)

    train.name.to_csv(file_paths[0], index=False, header=False)
    test.name.to_csv(file_paths[1], index=False, header=False)
    val.name.to_csv(file_paths[2], index=False, header=False)
