from typing import Tuple

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Input, Flatten, Lambda, Softmax, Layer, TimeDistributed
from tensorflow.keras.models import Model

from few_shot.dataset import OmniglotDataset

N_SHOT = 1
K_WAY = 60
N_QUERIES = 5

train_ds = OmniglotDataset('datasets/Omniglot/images_background', 1000000, N_SHOT, K_WAY, N_QUERIES)
train_it = train_ds.tf_iterator()

n_classes = train_ds.df.class_id.nunique()

# first retrieve embeddings
img_shape = (28, 28, 1)

def build_embedding_model(input_layer: Layer, n_convs=4):
    """Builds an embedding model as described in the Prototypical Networks paper."""
    embedding = input_layer  # need to keep a reference to the input
    for _ in range(n_convs):
        embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
        embedding = BatchNormalization()(embedding)
        embedding = ReLU()(embedding)
        embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Flatten()(embedding)
    model = Model(input_layer, embedding)

    return model


def centroids(support_embedding: Layer, support_labels: Layer, n_shot: int):
    """Computes class centroids as the class mean.

    :params
    support_embedding: layer containing the embeddings
    support_labels: one-hot encoded class labels
    n_shot: number of examples per class."""
    
    return tf.matmul(support_embedding, support_labels, transpose_a=True) / n_shot


def negative_distance(query_embedding: Layer, class_centroids: Layer):
    """Computes the negative squared euclidean distance between each query point and each class centroid."""

    # we need to expand the embedding tensor in order for broadcasting to work
    # our query tensor shape is (batch, q_queries, embedding_dims), and the centroids (batch, embedding_dims, k_way)
    # our output should be (batch, q_queries, embedding_dims, k_way)
    query_embedding = tf.expand_dims(query_embedding, axis=-1)
    sq_distance = tf.squared_difference(query_embedding, class_centroids)

    # reduce over the embedding dimension to find the distance between each query and class centroid
    # we return the negative distance since we want to activate the closest centroid, not the farthest (softmin?)
    return -tf.reduce_sum(sq_distance, axis=-2)


def build_prototype_network(n_shot, k_way, n_queries, input_shape, embedding_model_fn=build_embedding_model):
    embedding_in = Input(shape=input_shape)
    embedding_model = embedding_model_fn(embedding_in)
    
    support_in = Input(shape=(n_shot * k_way,) + input_shape, name='support_input')
    query_in = Input(shape=(n_queries * k_way,) + input_shape, name='query_input')
    support_labels = Input(shape=(n_shot * k_way, k_way), name='support_labels')

    # TimeDistributed is a convenient way to apply the same embedding model to high-dimensional batches
    support_embedding = TimeDistributed(embedding_model)(support_in)
    query_embedding = TimeDistributed(embedding_model)(query_in)
    
    # Lambda layers only accept a sequence of tensors as input
    class_centroids = Lambda(lambda x: centroids(*x, n_shot))((support_embedding, support_labels))
    
    negative_distances = Lambda(lambda x: negative_distance(*x))((query_embedding, class_centroids))
    
    predictions = Softmax()(negative_distances)
    
    model = Model(inputs=[support_in, support_labels, query_in], outputs=predictions)

    return model

model = build_prototype_network(N_SHOT, K_WAY, N_QUERIES, img_shape)

opt = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(train_it,
                    epochs=500,
                    steps_per_epoch=2000,
                    shuffle=False,
                    callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda i, lr: lr if i % 2000 else lr * 0.5)])
