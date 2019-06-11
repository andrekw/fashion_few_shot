from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from few_shot.dataset import FewShotEpisodeGenerator
from few_shot.dataset.fashion import build_fashion_df, TRAINING_CLASSES, TEST_CLASSES
from few_shot.model import build_embedding_model, build_prototype_network


def resize_img_pipeline_fn(img_shape):
    def resize_img_pipeline(path_tensor):
        img = tf.image.decode_image(tf.read_file(path_tensor),
                                    dtype=tf.float32,
                                    channels=img_shape[-1])

        return tf.image.resize_image_with_crop_or_pad(img, *img_shape[:-1])

    return resize_img_pipeline


def pad_validation_inputs(n_shot = 1,
                          n_queries_train = 5,
                          n_queries_test = 1,
                          k_way_train = 15,
                          k_way_test = 5):
    def pad_input(support_x, support_y, query_x, query_y):
        support_x = tf.pad(support_x,
                           [
                               [0, 0],
                               [0, n_shot * k_way_train - n_shot * k_way_test],
                               [0, 0],
                               [0, 0],
                               [0, 0]
                           ])
        support_y = tf.pad(support_y,
                           [
                               [0, 0],
                               [0, n_shot * k_way_train - n_shot * k_way_test],
                               [0, k_way_train - k_way_test]
                           ])
        query_x = tf.pad(query_x,
                         [
                             [0, 0],
                             [0, n_queries_train * k_way_train - n_queries_test * k_way_test],
                             [0, 0],
                             [0, 0],
                             [0, 0]
                         ])
        query_y = tf.pad(query_y,
                         [
                             [0, 0],
                             [0, n_queries_train * k_way_train - n_queries_test * k_way_test],
                             [0, k_way_train - k_way_test]
                         ])
        return support_x, support_y, query_x, query_y

    return pad_input


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


def evaluate_fashion_few_shot(train_df,
                              val_df,
                              test_df,
                              lr,
                              n_shot,
                              n_queries_train,
                              n_queries_test,
                              k_way_train,
                              eps_per_epoch,
                              n_epochs,
                              k_way_test,
                              test_eps,
                              img_shape):

    train_dataset = FewShotEpisodeGenerator(train_df[['class_name', 'filepath']].copy(),
                                            n_epochs * eps_per_epoch,
                                            n_shot,
                                            k_way_train,
                                            n_queries_train)

    val_dataset = FewShotEpisodeGenerator(val_df[['class_name', 'filepath']].copy(),
                                          n_epochs * eps_per_epoch,
                                          n_shot,
                                          k_way_test,
                                          n_queries_test)

    test_dataset = FewShotEpisodeGenerator(test_df[['class_name', 'filepath']].copy(),
                                           n_epochs * eps_per_epoch,
                                           n_shot,
                                           k_way_test,
                                           n_queries_test)

    train_it = train_dataset.tf_iterator(image_pipeline=resize_img_pipeline_fn(img_shape))
    val_it = val_dataset.tf_iterator(image_pipeline=resize_img_pipeline_fn(img_shape),
                                     post_transform=pad_validation_inputs(n_shot,
                                                                          n_queries_train,
                                                                          n_queries_test,
                                                                          k_way_train,
                                                                          k_way_test))

    embedding_input = tf.keras.layers.Input(shape=img_shape)
    embedding_model = build_embedding_model(embedding_input)
    model = build_prototype_network(n_shot,
                                    k_way_train,
                                    n_queries_train,
                                    img_shape,
                                    embedding_model_fn=lambda x: embedding_model)

    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(train_it,
              epochs=n_epochs,
              steps_per_epoch=eps_per_epoch,
              shuffle=False,
              validation_data=val_it,
              validation_steps=eps_per_epoch,
              callbacks=[
                  tf.keras.callbacks.LearningRateScheduler(
                      lambda i, lr: lr if i % (2000//eps_per_epoch) else lr * 0.5)])
    test_it = test_dataset.tf_iterator(image_pipeline=resize_img_pipeline_fn(img_shape))

    test_model = build_prototype_network(n_shot,
                                         k_way_test,
                                         n_queries_test,
                                         img_shape,
                                         embedding_model_fn=lambda x: embedding_model)
    test_opt = tf.keras.optimizers.Adam(lr=lr)
    test_model.compile(optimizer=test_opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    test_loss, test_acc = test_model.evaluate(test_it, steps=test_eps)
    return test_loss, test_acc