import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

import few_shot.experiments.fashion.config as config
from few_shot.experiments.fashion import evaluate_fashion_few_shot
from few_shot.dataset.fashion import fashion_dfs
from few_shot.dataset.image_pipeline import augmented_img_pipeline_fn, class_augmentation_fn

if __name__ == '__main__':
    np.random.seed(23)
    tf.random.set_random_seed(29)
    train_df, val_df, test_df = fashion_dfs()

    results = []
    for n_shots, k_way_test in itertools.product(config.SHOTS, config.TEST_K_WAY):
        print(f'Running fashion experiment {n_shots}-shot, {k_way_test} way')
        assert k_way_test <= test_df.class_name.nunique()
        result = evaluate_fashion_few_shot(train_df=train_df,
                                           val_df=val_df,
                                           test_df=test_df,
                                           n_shot=n_shots,
                                           k_way_test=k_way_test,
                                           patience=20,
                                           restore_best_weights=True,
                                           img_pipeline_fn=augmented_img_pipeline_fn,
                                           post_processing_fn=class_augmentation_fn(config.K_WAY_TRAIN))

        results.append(result)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv('fashion_class_augmentation_plus_old_augmentation_results.csv')
