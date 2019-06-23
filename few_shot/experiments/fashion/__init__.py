import sys
from typing import Tuple, Callable, List

import pandas as pd
import tensorflow as tf

from . import config
from few_shot.dataset import FewShotEpisodeGenerator
from few_shot.dataset.image_pipeline import resize_img_pipeline_fn
from few_shot.model import build_embedding_model, build_prototype_network


def evaluate_fashion_few_shot(train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              n_shot: int,
                              k_way_test: int,
                              lr: float = config.lr,
                              n_queries_train: int = config.N_QUERIES_TRAIN,
                              n_queries_test: int = config.N_QUERIES_TEST,
                              k_way_train: int = config.K_WAY_TRAIN,
                              eps_per_epoch: int = config.EPS_PER_EPOCH,
                              n_epochs: int = config.N_EPOCHS,
                              test_eps: int = config.TEST_EPS,
                              img_shape: Tuple[int, int, int] = config.IMG_SHAPE,
                              img_pipeline_fn:
                              Callable[[Tuple[int, int, int]], Callable[[str], tf.Tensor]] = resize_img_pipeline_fn,
                              patience: int = config.PATIENCE,
                              opt: tf.keras.optimizers.Optimizer = None,
                              callbacks: List[tf.keras.callbacks.Callback] = None,
                              restore_best_weights: bool = True,
                              embedding_fn:
                              Callable[[tf.keras.layers.Layer], tf.keras.models.Model] = build_embedding_model,
                              reduce_lr_on_plateau: bool = False,
                              reduction_factor: float = 0.75,
                              validation_metric: str = 'loss',
                              post_processing_fn:
                              Callable[[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                                       Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = None,
                              augment: bool = False):
    """
    Train and evaluates a Prototypical Network on a given dataset.

    :param train_df: DataFrame with training rows.
    :param val_df: DataFrame with validation rows.
    :param test_df: DataFrame with evaluation rows.
    :param n_shot: Examples per class in the support set.
    :param k_way_test: Number of classes per evaluation episode.
    :param lr: initial learning rate.
    :param n_queries_train: Number of examples in the query set for training.
    :param n_queries_test: Number of examples in the query set for validation and evaluation.
    :param k_way_train: Number of classes in each evaluation episode.
    :param eps_per_epoch: How many episodes to count as an epoch, for validation and scheduling purposes.
    :param n_epochs: Maximum number of epochs to train for.
    :param test_eps: How many episodes to evaluate on.
    :param img_shape: dimensions of each image in the experiment.
    :param img_pipeline_fn: a function that takes a filename as an input and return an image tensor.
    :param patience: how many episodes to wait for before stopping early.
    :param opt: Optimizer to use.
    :param callbacks: Callbacks for the fit function.
    :param restore_best_weights: Whether to use best-validated or latest weights after stopping training.
    :param embedding_fn: a function that takes a keras layer as an input and returns an embedding model from that input.
    :param reduce_lr_on_plateau: Whether to reduce the learning rate if validation loss does not drop.
    :param reduction_factor: multiplier for the learning rate when at a plateau.
    :param validation_metric: one of {'loss', 'accuracy'}. Which metric to base early stopping on.
    :param post_processing_fn: a function to process episode data before training or testing. Used i.e. for class
    augmentation.
    :param augment: Whether to use the TF_hub-based augmentation layer.
    :return:
    """
    args = locals()
    args.pop('train_df')
    args.pop('test_df')
    args.pop('val_df')
    args.pop('img_pipeline_fn')
    args.pop('opt')
    args.pop('callbacks')
    print(args)

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

    train_it = train_dataset.tf_iterator(image_pipeline=img_pipeline_fn(img_shape),
                                         post_transform=post_processing_fn)
    val_it = val_dataset.tf_iterator(image_pipeline=resize_img_pipeline_fn(img_shape))

    embedding_input = tf.keras.layers.Input(shape=img_shape)
    embedding_model = embedding_fn(embedding_input)
    model = build_prototype_network(n_shot,
                                    k_way_train,
                                    img_shape,
                                    augment=augment,
                                    embedding_model_fn=lambda x: embedding_model)

    if not opt:
        opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    if not callbacks:
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(
                lambda i, lr: lr if not i or i % (2000//eps_per_epoch) else lr * 0.5, verbose=1),
              ]

    test_it = test_dataset.tf_iterator(image_pipeline=resize_img_pipeline_fn(img_shape))

    test_model = build_prototype_network(n_shot,
                                         k_way_test,
                                         img_shape,
                                         embedding_model_fn=lambda x: embedding_model)
    test_opt = tf.keras.optimizers.Adam(lr=lr)
    test_model.compile(optimizer=test_opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    best_val_acc = 0.0
    best_val_loss = sys.float_info.max
    best_weights = model.get_weights()
    curr_step = 0  # for patience

    for i in range(n_epochs):
        print('Training:')
        history = model.fit(train_it,
                            epochs=i + 1,
                            initial_epoch=i,
                            steps_per_epoch=eps_per_epoch,
                            shuffle=False,
                            callbacks=callbacks,
                            verbose=1)

        print(history.history)
        latest_weights = model.get_weights()

        print('Validation:')
        val_loss, val_acc = test_model.evaluate(val_it, steps=eps_per_epoch)
        print(f'epoch {i}: val_loss: {val_loss}, val_cat_accuracy: {val_acc}')
        if validation_metric == 'loss' and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            curr_step = 0
        elif validation_metric == 'accuracy' and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            curr_step = 0
        else:
            curr_step += 1
            if reduce_lr_on_plateau:
                new_lr = max(tf.keras.backend.get_value(model.optimizer.lr) * reduction_factor, 1e-4)
                print(f'reduced lr to {new_lr}')
                tf.keras.backend.set_value(
                    model.optimizer.lr,
                    new_lr)

        if curr_step > patience:
            break
    if restore_best_weights:
        test_model.set_weights(best_weights)
        for a, b in zip(best_weights, test_model.get_weights()):
            assert (a == b).all()
    test_loss, test_acc = test_model.evaluate(test_it, steps=test_eps)
    args.update({
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_batches': i
    })
    return args
