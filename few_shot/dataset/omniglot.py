import os

import pandas as pd


def create_omniglot_df(path: str) -> pd.DataFrame:
    """Creates a DataFrame for Omniglot data based on its directory structure.

    :param path: path to the base folder of the processed Omniglot dataset.
    """
    samples = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not filenames:
            continue
        *_, alphabet, character = dirpath.split(os.path.sep)
        class_name = f'{alphabet}.{character}'
        for f in filenames:
            samples.append(
                {
                    'class_name': class_name,
                    'alphabet': alphabet,
                    'character': character,
                    'filepath': os.path.abspath(os.path.join(dirpath, f))
                })

    return pd.DataFrame.from_records(samples)