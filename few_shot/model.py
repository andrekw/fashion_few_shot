import tensorflow as tf

from tensorflow.keras.layers import (Conv2D, BatchNormalization, ReLU, MaxPooling2D,
                                     Input, Flatten, Lambda, Softmax, Layer, TimeDistributed, SpatialDropout2D)
from tensorflow.keras.models import Model


N_SHOT = 1
K_WAY = 60
N_QUERIES = 5


def build_embedding_model(input_layer: Layer, n_convs: int = 4, dropout: float = 0.0):
    """Builds an embedding model as described in the Prototypical Networks paper."""
    embedding = input_layer  # need to keep a reference to the input
    for _ in range(n_convs):
        embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
        embedding = BatchNormalization()(embedding)
        embedding = ReLU()(embedding)
        if dropout:
            embedding = SpatialDropout2D(dropout)(embedding)
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
    # our output should be (batch, q_queries, k_way, embedding_dims)
    query_embedding = tf.expand_dims(query_embedding, axis=-2)
    # we reshape the centroids to make broadcasting work
    class_centroids = tf.expand_dims(tf.transpose(class_centroids, [0, 2, 1]), axis=-3)
    sq_distance = tf.squared_difference(query_embedding, class_centroids)

    # reduce over the embedding dimension to find the distance between each query and class centroid
    # we return the negative distance since we want to activate the closest centroid, not the farthest (softmin?)
    return -tf.reduce_sum(sq_distance, axis=-1)


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
