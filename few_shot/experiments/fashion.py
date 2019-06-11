from typing import Tuple

import numpy as np
import pandas as pd

from few_shot.dataset.fashion import build_fashion_df, TRAINING_CLASSES, TEST_CLASSES


def fashion_dfs(dataset_path: str, n_val_classes: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds train, validation and test DataFrames from the kaggle fashion dataset.

    :param dataset_path: path to the dataset
    :param n_val_classes: how many classes to use in the validation set
    :returns: a tuple of train, validation and test DataFrames
    """
    df = build_fashion_df(dataset_path)

    val_classes = set(np.random.choice(list(TRAINING_CLASSES), n_val_classes, replace=False))
    train_df = df[df.class_name.isin(TRAINING_CLASSES - val_classes)]
    val_df = df[df.class_name.isin(val_classes)]

    test_df = df[df.class_name.isin(TEST_CLASSES)]

    return train_df, val_df, test_df
