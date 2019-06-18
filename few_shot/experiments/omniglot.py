import numpy as np
import tensorflow as tf

from few_shot.dataset import FewShotEpisodeGenerator
from few_shot.dataset.omniglot import create_omniglot_df
from few_shot.model import build_embedding_model, build_prototype_network


tf.enable_eager_execution()  # tf.data has issues with keras in graph mode


def run_omniglot_experiment():
    # set seed so we get reproducible results
    np.random.seed(23)
    tf.random.set_random_seed(29)

    lr = 1e-3
    n_shot = 1
    n_queries_train = 5
    n_queries_test = 1
    k_way_train = 60
    eps_per_epoch = 100
    n_epochs = 40

    k_way_test = 5
    test_eps = 1000

    train_ds = FewShotEpisodeGenerator(create_omniglot_df('datasets/Omniglot/images_background'),
                                       1000000,
                                       n_shot,
                                       k_way_train,
                                       n_queries_train)
    train_it = train_ds.tf_iterator()

    test_ds = FewShotEpisodeGenerator(create_omniglot_df('datasets/Omniglot/images_evaluation'),
                                      1000000,
                                      n_shot,
                                      k_way_test,
                                      n_queries_test)
    test_it = test_ds.tf_iterator()

    img_shape = (28, 28, 1)
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
              callbacks=[
                  tf.keras.callbacks.LearningRateScheduler(
                      lambda i, lr: lr if not i or i % (2000//eps_per_epoch) else lr * 0.5),
                  tf.keras.callbacks.TensorBoard(
                      log_dir=f'experiments/logs/omniglot_lr={lr}_n={n_shot}_k={k_way_train}_q={n_queries_train}')
              ])

    # since we changed the dimension of the inputs, we reuse weights in a new model
    test_model = build_prototype_network(n_shot,
                                         k_way_test,
                                         n_queries_test,
                                         img_shape,
                                         embedding_model_fn=lambda x: embedding_model)

    test_opt = tf.keras.optimizers.Adam(lr=lr)
    test_model.compile(optimizer=test_opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    test_loss, test_acc = test_model.evaluate(test_it, steps=test_eps)

    print(f'5-way, 1-shot test: loss: {test_loss}, categorical accuracy: {test_acc}')


if __name__ == '__main__':
    run_omniglot_experiment()
