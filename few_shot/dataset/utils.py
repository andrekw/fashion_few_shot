import tensorflow as tf


def perturb_image(X, p, flipx=True, flipy=True, scale=1.2, rot=3.14/4, translate=0.8, is_training=False):
    shape = tf.shape(X)

    width = tf.cast(shape[0], tf.float32)
    height = tf.cast(shape[1], tf.float32)
    channels = shape[2]

    identity = [1.0, 0, 0, 0, 1, 0, 0, 0]

    transforms = [identity]

    if flipx and tf.random_uniform([]) < p:
        X = tf.image.flip_left_right(X)
    if flipy and tf.random_uniform([]) < p:
        X = tf.image.flip_up_down(X)
    scale = tf.random_uniform([], minval=1.0/scale, maxval=scale)
    transforms.append(scale * identity)

    rads = tf.random_uniform([], minval=-rot/2.0, maxval=rot/2.0)
    transforms.append(tf.contrib.image.angles_to_projective_transforms(rads, height, width))

    translations = tf.contrib.image.translations_to_projective_transforms(
        tf.random_uniform([2], minval=-translate/2, maxval=translate/2))
    transforms.append(translations)

    brightened = tf.image.random_brightness(X, 0.5)
    contrasted = tf.image.random_contrast(X, 0.2, 0.5)

    return tf.cond(is_training,
                   lambda: tf.contrib.image.transform(
                       brightened, 
                       tf.contrib.image.compose_transforms(*transforms))
                       + tf.random_normal(X.shape, stddev=0.01),
                   lambda: X)