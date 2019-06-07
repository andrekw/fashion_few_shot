import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Input, Flatten, Lambda
from tensorflow.keras.models import Model

from .dataset import OmniglotDataset

N_SHOT = 1
K_WAY = 60
N_QUERIES = 5

ds = OmniglotDataset('datasets/Omniglot', 100, N_SHOT, K_WAY, N_QUERIES)
it = ds.tf_iterator()

n_classes = ds.df.class_id.nunique()

# first retrieve embeddings
embedding_input = Input(shape=(28, 28, 1))

embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding_input)
embedding = BatchNormalization()(embedding)
embedding = ReLU()(embedding)
embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
embedding = BatchNormalization()(embedding)
embedding = ReLU()(embedding)
embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
embedding = BatchNormalization()(embedding)
embedding = ReLU()(embedding)
embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
embedding = Conv2D(64, 3, data_format='channels_last', padding='same')(embedding)
embedding = BatchNormalization()(embedding)
embedding = ReLU()(embedding)
embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
embedding = Flatten()(embedding)
embedding_model = Model(embedding_input, embedding)

support_in = Input(shape=(28, 28, 1), name='support_input')
query_in = Input(shape=(28, 28, 1), name='query_input')
support_labels = Input(shape=(n_classes,), name='support_labels')

support_embedding = embedding_model(support_in)
query_embedding = embedding_model(query_in)

class_centroids = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True) / N_SHOT)([support_embedding, support_labels])

distances = Lambda(lambda x: tf.reduce_sum(
    tf.squared_difference(tf.expand_dims(x[0], axis=-1), x[1]), axis=1))([query_embedding, class_centroids])

model = Model(inputs=[support_in, support_labels, query_in], outputs=distances)

opt = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
