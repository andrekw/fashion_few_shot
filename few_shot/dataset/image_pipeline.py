import tensorflow as tf


def perturb_image(X, p, flipx=True, flipy=True, scale=1.2, rot=3.14/4, translate=0.8, is_training=False):
    p = tf.convert_to_tensor(p)
    is_training = tf.convert_to_tensor(is_training)
    shape = tf.shape(X)

    width = tf.cast(shape[0], tf.float32)
    height = tf.cast(shape[1], tf.float32)

    identity = [1.0, 0, 0, 0, 1, 0, 0, 0]

    transforms = [identity]

    if flipx:
        X = tf.cond(tf.random_uniform([]) < p, lambda: tf.image.flip_left_right(X), lambda: X)
    if flipy:
        X = tf.cond(tf.random_uniform([]) < p, lambda: tf.image.flip_up_down(X), lambda: X)
    scale = tf.random_uniform([], minval=1.0/scale, maxval=scale)
    transforms.append(scale * identity)

    rads = tf.random_uniform([], minval=-rot/2.0, maxval=rot/2.0)
    transforms.append(tf.contrib.image.angles_to_projective_transforms(rads, height, width))

    translations = tf.contrib.image.translations_to_projective_transforms(
        tf.random_uniform([2], minval=-translate/2, maxval=translate/2))
    transforms.append(translations)

    brightened = tf.image.random_brightness(X, 0.1)
    # contrasted = tf.image.random_contrast(brightened, 0.2, 0.3)

    return tf.cond(is_training,
                   lambda: tf.contrib.image.transform(
                       brightened,
                       tf.contrib.image.compose_transforms(*transforms), interpolation='BILINEAR')
                   + tf.random_normal(X.shape, stddev=0.01),
                   lambda: X)


def augmented_img_pipeline_fn(img_shape):
    def resize_img_pipeline(path_tensor):
        img = tf.image.decode_image(tf.read_file(path_tensor),
                                    dtype=tf.float32,
                                    channels=img_shape[-1])
        img = tf.image.resize_image_with_pad(img, *img_shape[:-1])
        img.set_shape(img_shape)
        img = perturb_image(img, 0.5, is_training=True, translate=1, flipy=False, rot=3.14/6)
        return tf.clip_by_value(img, 0, 1)

    return resize_img_pipeline


def resize_img_pipeline_fn(img_shape):
    def resize_img_pipeline(path_tensor):
        img = tf.image.decode_image(tf.read_file(path_tensor),
                                    dtype=tf.float32,
                                    channels=img_shape[-1])

        return tf.image.resize_image_with_pad(img, *img_shape[:-1])

    return resize_img_pipeline
