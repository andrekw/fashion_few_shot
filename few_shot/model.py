import tensorflow as tf
import tensorflow.contrib.image
import tensorflow_hub as hub

from tensorflow.keras.layers import (Conv2D, BatchNormalization, ReLU, MaxPooling2D,
                                     Input, Flatten, Lambda, Softmax, Layer, SpatialDropout2D)
from tensorflow.keras.models import Model


N_SHOT = 1
K_WAY = 60
N_QUERIES = 5


class AugLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AugLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.augmentation_module = hub.Module(
            'https://tfhub.dev/google/image_augmentation/flipx_crop_rotate_color/1')
        super(AugLayer, self).build(input_shape)

    def call(self, x, training=None):
        params = {
            'images': x,
            'image_size': self.output_dim,
            'augmentation': bool(True)
        }
        return self.augmentation_module(params, signature='from_decoded_images')


def build_embedding_model(input_layer: Layer, n_convs: int = 4, dropout: float = 0.0):
    """Builds an embedding model as described in the Prototypical Networks paper."""
    embedding = input_layer  # need to keep a reference to the input
    for _ in range(n_convs):
        embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
        embedding = BatchNormalization(momentum=0.9, epsilon=1e-8)(embedding)
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
    class_centroids = tf.expand_dims(tf.linalg.transpose(class_centroids), axis=-3)
    sq_distance = tf.squared_difference(query_embedding, class_centroids)

    # reduce over the embedding dimension to find the distance between each query and class centroid
    # we return the negative distance since we want to activate the closest centroid, not the farthest (softmin?)
    return -tf.reduce_sum(sq_distance, axis=-1)


def build_prototype_network(n_shot, k_way, n_queries, input_shape, embedding_model_fn=build_embedding_model,
                            augment=False):
    """Builds a prototype network based on an image embedding module."""
    embedding_in = Input(shape=input_shape)
    if augment:
        aug_layer = AugLayer(input_shape[:-1])
        embedding_in = aug_layer(embedding_in)
    embedding_model = embedding_model_fn(embedding_in)

    support_in = Input(shape=input_shape, name='support_input')
    query_in = Input(shape=input_shape, name='query_input')
    support_labels = Input(shape=(k_way,), name='support_labels')

    support_embedding = embedding_model(support_in)
    query_embedding = embedding_model(query_in)

    # Lambda layers only accept a sequence of tensors as input
    class_centroids = Lambda(lambda x: centroids(*x, n_shot))((support_embedding, support_labels))

    negative_distances = Lambda(lambda x: negative_distance(*x))((query_embedding, class_centroids))

    predictions = Softmax()(negative_distances)

    model = Model(inputs=[support_in, support_labels, query_in], outputs=predictions)

    return model
