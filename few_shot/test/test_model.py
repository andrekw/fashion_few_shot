import pytest
import tensorflow as tf

from few_shot.model import build_prototype_network, centroids


tf.enable_eager_execution()


"""
Since we are potentially dealing with a large number of floating-point operations,
we are at risk of accumulating enough error to make our computed and expected tensors
different enough to fail tests.
"""
TEST_DTYPE = tf.float64


@pytest.mark.parametrize("eps_per_batch", [1, 4])
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
                class_samples = tf.random.normal((n_shot, embedding_dims), mean=j)
                class_labels = [j] * n_shot
                class_centroid = tf.reduce_sum(class_samples, axis=0) / n_shot

                ep_samples.append(class_samples)
                ep_labels.append(class_labels)
                ep_prototypes.append(class_centroid)

            samples.append(tf.concat(ep_samples, 0))
            labels.append(tf.concat(ep_labels, 0))
            prototypes.append(tf.stack(ep_prototypes))

        expected_centroids = tf.stack(prototypes, 0)
        samples = tf.stack(samples, 0)
        labels = tf.stack(tf.one_hot(labels, n_classes), 0)

        result = centroids(samples, n_shot)

        assert pytest.approx(expected_centroids.numpy()) == result.numpy()
        # check shape as well: one centroid per class per episode
        assert result.shape.dims == [eps_per_batch, n_classes, embedding_dims]


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('k', [5, 60])
@pytest.mark.parametrize('q', [1, 5])
@pytest.mark.parametrize('img_shape', [(28, 28, 1), (40, 30, 3)])
def test_model(n, k, q, img_shape):
    model = build_prototype_network(n, k, q, img_shape)
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # we have q * k query points (q per class) and we compare its distance to each class centroid
    assert model.output_shape == (None, q * k, k)


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('k', [5, 60])
@pytest.mark.parametrize('q', [1, 5])
@pytest.mark.parametrize('embedding_dims', [64, 128])
def test_negative_distance(n, k, q, embedding_dims):
    with tf.device('CPU:0'):
        pass
