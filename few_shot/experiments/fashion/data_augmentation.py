import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from few_shot.experiments.fashion import evaluate_fashion_few_shot, fashion_dfs
from few_shot.dataset.utils import perturb_image


def augmented_img_pipeline_fn(img_shape):
    def resize_img_pipeline(path_tensor):
        img = tf.image.decode_image(tf.read_file(path_tensor),
                                    dtype=tf.float32,
                                    channels=img_shape[-1])
        img = tf.image.resize_image_with_pad(img, *img_shape[:-1])
        img.set_shape(img_shape)
        return perturb_image(img, 0.5, is_training=True, translate=1, flipy=False, rot=3.14/6)

    return resize_img_pipeline


if __name__ == '__main__':
    tf.enable_eager_execution()

    np.random.seed(23)
    tf.random.set_random_seed(29)

    SHOTS = [5, 1]
    TEST_K_WAY = [15, 5]

    lr = 1e-3
    n_queries_train = 5
    n_queries_test = 5
    k_way_train = 20
    eps_per_epoch = 100
    n_epochs = 100
    test_eps = 1000
    img_shape = (160, 120, 3)  # in order to be able to fit everything in memory with a large k-way

    train_df, val_df, test_df = fashion_dfs('datasets/fashion-dataset',
                                            min_rows=n_queries_train + max(SHOTS),  # support and query
                                            n_val_classes=16)

    assert k_way_train <= train_df.class_name.nunique()
    assert 16 == val_df.class_name.nunique()

    results = []
    for n_shots, k_way_test in itertools.product(SHOTS, TEST_K_WAY):
        print(f'Running fashion experiment {n_shots}-shot, {k_way_test} way')
        assert k_way_test <= test_df.class_name.nunique()
        result = evaluate_fashion_few_shot(train_df=train_df,
                                           val_df=val_df,
                                           test_df=test_df,
                                           lr=lr,
                                           n_shot=n_shots,
                                           n_queries_train=n_queries_train,
                                           n_queries_test=n_queries_test,
                                           k_way_train=k_way_train,
                                           eps_per_epoch=eps_per_epoch,
                                           n_epochs=n_epochs,
                                           k_way_test=k_way_test,
                                           test_eps=test_eps,
                                           img_shape=img_shape,
                                           img_pipeline_fn=augmented_img_pipeline_fn)

        results.append(result)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv('fashion_augmentation_only_results.csv')
