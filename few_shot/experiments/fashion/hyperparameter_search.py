import itertools

import numpy as np
import pandas as pd
import skopt
import tensorflow as tf

from few_shot.experiments.fashion import evaluate_fashion_few_shot, fashion_dfs
from few_shot.dataset.image_pipeline import augmented_img_pipeline_fn

MAX_SHOTS = 5


def few_shot_optimize(train_df,
                      val_df,
                      test_df,
                      n_shot,
                      n_queries_train,
                      n_queries_test,
                      eps_per_epoch,
                      n_epochs,
                      k_way_test,
                      test_eps,
                      img_shape):

    experiment_val_classes = set(np.random.choice(train_df.class_name.unique(), size=16, replace=False))
    experiment_train_df = train_df[~train_df.class_name.isin(experiment_val_classes)]
    experiment_val_df = train_df[train_df.class_name.isin(experiment_val_classes)]

    dimensions = [
        skopt.space.Categorical(name='optimizer_type', categories=('adam', 'rmsprop')),
        skopt.space.Real(name='learning_rate', low=1e-3, high=3),
        skopt.space.Categorical(name='k_way_train_type', categories=('large', 'same')),
        skopt.space.Integer(name='early_stop_patience', low=1, high=5)
        ]

    @skopt.utils.use_named_args(dimensions)
    def evaluate_parameters(optimizer_type, learning_rate, k_way_train_type, early_stop_patience):
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
        else:
            raise ValueError('Unsupported optimizer_type')

        if k_way_train_type == 'large':
            cur_k_train = k_way_train
        elif k_way_train_type == 'same':
            cur_k_train = k_way_test
        else:
            raise ValueError('Unsupported k value')
        img_fn = augmented_img_pipeline_fn

        result = evaluate_fashion_few_shot(train_df=experiment_train_df,
                                           val_df=experiment_val_df,
                                           test_df=val_df,
                                           lr=lr,
                                           n_shot=n_shots,
                                           n_queries_train=n_queries_train,
                                           n_queries_test=n_queries_test,
                                           k_way_train=cur_k_train,
                                           eps_per_epoch=eps_per_epoch,
                                           n_epochs=n_epochs,
                                           k_way_test=k_way_test,
                                           test_eps=test_eps,
                                           img_shape=img_shape,
                                           opt=optimizer,
                                           img_pipeline_fn=img_fn)
        result['optimizer'] = optimizer_type

        return result['test_loss']

    res = skopt.gp_minimize(evaluate_parameters, dimensions, n_calls=10, n_random_starts=5)

    best_opt, best_lr, best_k_way_type, best_patience = res.x

    print(res.x)

    if best_opt == 'adam':
        opt = tf.keras.optimizers.Adam(lr=best_lr)
    elif best_opt == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(lr=best_lr)
    else:
        raise ValueError('oops')

    result = evaluate_fashion_few_shot(train_df=train_df,
                                       val_df=val_df,
                                       test_df=test_df,
                                       lr=lr,
                                       n_shot=n_shots,
                                       n_queries_train=n_queries_train,
                                       n_queries_test=n_queries_test,
                                       k_way_train=k_way_train if best_k_way_type == 'large' else k_way_test,
                                       eps_per_epoch=eps_per_epoch,
                                       n_epochs=n_epochs,
                                       k_way_test=k_way_test,
                                       test_eps=test_eps,
                                       img_shape=img_shape,
                                       opt=opt,
                                       patience=best_patience,
                                       img_pipeline_fn=augmented_img_pipeline_fn)
    result['opt'] = best_opt

    return result


if __name__ == '__main__':
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
        result = few_shot_optimize(train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   n_shot=n_shots,
                                   n_queries_train=n_queries_train,
                                   n_queries_test=n_queries_test,
                                   eps_per_epoch=eps_per_epoch,
                                   n_epochs=n_epochs,
                                   k_way_test=k_way_test,
                                   test_eps=test_eps,
                                   img_shape=img_shape)

        results.append(result)

    df = pd.DataFrame.from_records(results)
    print(df)
    df.to_csv('fashion_hyperparameter_search_results.csv')
