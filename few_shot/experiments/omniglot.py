import tensorflow as tf
tf.enable_eager_execution()  # tf.data has issues with keras in graph mode

from few_shot.dataset import OmniglotDataset
from few_shot.model import build_embedding_model, build_prototype_network

n_shot = 1
n_queries = 5
k_way_train = 60

k_way_test = 5


train_ds = OmniglotDataset('datasets/Omniglot/images_background', 1000000, n_shot, k_way_train, n_queries)
train_it = train_ds.tf_iterator()

test_ds = OmniglotDataset('datasets/Omniglot/images_evaluation', 1000, n_shot, k_way_test, n_queries)
test_it = test_ds.tf_iterator()

img_shape = (28, 28, 1)
embedding_input = tf.keras.layers.Input(shape=img_shape)
embedding_model = build_embedding_model(embedding_input)

model = build_prototype_network(n_shot, k_way_train, n_queries, img_shape, embedding_model_fn=lambda x: embedding_model)

opt = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(train_it,
                    epochs=50,
                    steps_per_epoch=200,
                    shuffle=False,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda i, lr: lr if i % 2000 else lr * 0.5)])
#                    validation_data=val_it, validation_steps=60)
