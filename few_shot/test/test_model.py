import pytest
import tensorflow as tf

from few_shot.model import build_prototype_network, centroids, negative_distance


tf.enable_eager_execution()


"""
Since we are potentially dealing with a large number of floating-point operations,
we are at risk of accumulating enough error to make our computed and expected tensors
different enough to fail tests.
"""
TEST_DTYPE = tf.float64


@pytest.mark.parametrize('eps_per_batch', [1, 4])
@pytest.mark.parametrize("n_shot", [1, 5])
@pytest.mark.parametrize("n_classes", [5, 15, 60])
@pytest.mark.parametrize("embedding_dims", [60, 300])
def test_centroids(eps_per_batch, n_shot, n_classes, embedding_dims):
    samples = []
    labels = []
    prototypes = []

    with tf.device('CPU:0'):

        # build random samples and the expected result

        for i in range(eps_per_batch):
            ep_samples = []
            ep_labels = []
            ep_prototypes = []
            for j in range(n_classes):
                class_samples = tf.random.normal((n_shot, embedding_dims), mean=j, dtype=TEST_DTYPE)
                class_labels = [j] * n_shot
                class_centroid = tf.reduce_sum(class_samples, axis=0) / n_shot

                ep_samples.append(class_samples)
                ep_labels.append(class_labels)
                ep_prototypes.append(class_centroid)

            samples.append(tf.concat(ep_samples, 0))
            labels.append(tf.concat(ep_labels, 0))
            prototypes.append(tf.stack(ep_prototypes))

        expected_centroids = tf.transpose(tf.stack(prototypes, 0), [0, 2, 1])
        samples = tf.stack(samples, 0)
        labels = tf.stack(tf.one_hot(labels, n_classes, dtype=TEST_DTYPE), 0)

        result = centroids(samples, labels, n_shot)

        assert pytest.approx(expected_centroids.numpy()) == result.numpy()
        # check shape as well: one centroid per class per episode
        assert result.shape.dims == [eps_per_batch, embedding_dims, n_classes]


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('k', [5, 60])
@pytest.mark.parametrize('img_shape', [(28, 28, 1), (40, 30, 3)])
def test_model(n, k, img_shape):
    model = build_prototype_network(n, k, img_shape)
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # we have a dynamic number of query points (q per class) and we compare its distance to each class centroid
    assert model.output_shape == (None, k)


@pytest.mark.parametrize('eps_per_batch', [1, 5])
@pytest.mark.parametrize('k', [5, 60])
@pytest.mark.parametrize('q', [1, 5])
@pytest.mark.parametrize('embedding_dims', [64, 128])
def test_negative_distance(eps_per_batch, k, q, embedding_dims):
    with tf.device('CPU:0'):
        class_centroids = []
        queries = []
        expected_distances = []

        for _ in range(eps_per_batch):
            ep_centroids = tf.random.normal((k, embedding_dims), stddev=1.5, dtype=TEST_DTYPE)
            ep_queries = []
            ep_expected_distances = []

            for _ in range(q * k):
                query_distances = []
                query_embedding = tf.random.normal((embedding_dims,), stddev=2.0, dtype=TEST_DTYPE)
                for i in range(k):  # compare it to each centroid
                    query_expected_distance = tf.reduce_sum((query_embedding - ep_centroids[i])**2)
                    query_distances.append(query_expected_distance)
                ep_queries.append(query_embedding)
                ep_expected_distances.append(tf.stack(query_distances, 0))
            class_centroids.append(ep_centroids)
            queries.append(tf.stack(ep_queries, 0))
            expected_distances.append(tf.stack(ep_expected_distances, 0))

        class_centroids = tf.transpose(tf.stack(class_centroids), [0, 2, 1])
        queries = tf.stack(queries)
        expected_distances = tf.stack(expected_distances)

        distances = negative_distance(queries, class_centroids)
        assert distances.shape.dims == [eps_per_batch, q * k, k]
        assert pytest.approx(-expected_distances.numpy()) == distances.numpy()
