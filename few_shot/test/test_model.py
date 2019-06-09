import pytest
import tensorflow as tf
tf.enable_eager_execution()

from few_shot.model import centroids

def test_centroids():
    embedding_dims = 4
    n_classes = 5
    n_shot = 1
    eps_per_batch = 1

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
                class_labels = [j]
                class_centroid = tf.reduce_sum(class_samples, axis=0) / n_shot

                ep_samples.append(class_samples)
                ep_labels.append(class_labels)
                ep_prototypes.append(class_centroid)

            samples.append(tf.concat(ep_samples, 0))
            labels.append(tf.concat(ep_labels, 0))
            prototypes.append(tf.stack(ep_prototypes, 1))  # expected transposed

        expected_centroids = tf.stack(prototypes, 0)
        samples = tf.stack(samples, 0)
        labels = tf.stack(tf.one_hot(labels, n_classes), 0)

        result = centroids(samples, labels, n_shot)

        assert pytest.approx(result.numpy()) == expected_centroids.numpy()
        # check shape as well: one centroid per class per episode (class v. embedding transposed)
        assert result.shape.dims == [eps_per_batch, embedding_dims, n_classes]
