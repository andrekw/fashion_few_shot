import os

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import tensorflow as tf


def create_omniglot_df(path: str) -> pd.DataFrame:
    """Creates a DataFrame for Omniglot data based on its directory structure.

    :param path: path to the base folder of the processed Omniglot dataset.
    """
    samples = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not filenames:
            continue
        *_, alphabet, character = dirpath.split(os.path.sep)
        class_name = f'{alphabet}.{character}'
        for f in filenames:
            samples.append(
                {
                    'class_name': class_name,
                    'alphabet': alphabet,
                    'character': character,
                    'filepath': os.path.abspath(os.path.join(dirpath, f))
                })

    return pd.DataFrame.from_records(samples)


class FewShotEpisodeGenerator(object):

    def __init__(self, df: pd.DataFrame, episodes: int, n_shot: int, k_way: int, q_queries: int, batch_size: int = 1):
        """Encapsulates the logic to build few-shot episodes from the Omniglot dataset.

        :param df: DataFrame with class_name and filepath columns
        :param episodes: episodes to generate
        :param n_shot: samples per class in support set
        :param k_way: classes in both support and query sets
        :param q_queries: samples per class in query set
        :param batch_size: episodes per batch
        """

        self.df = df
        self.episodes = episodes
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.batch_size = batch_size

        self.df['class_id'] = skp.LabelEncoder().fit_transform(self.df.class_name)

    def __iter__(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        for _ in range(self.episodes):
            classes = np.random.choice(self.df.class_id.unique(), size=self.k_way, replace=False)
            support_X = []
            support_y = []
            query_X = []
            query_y = []

            for k in classes:
                k_support = self.df[self.df.class_id == k].sample(self.n_shot)
                ks_idx = k_support.index.values  # index of support samples
                k_query = self.df[(self.df.class_id == k) & (~self.df.index.isin(ks_idx))].sample(self.q_queries)

                support_X += k_support['filepath'].tolist()
                support_y += k_support['class_id'].tolist()

                query_X += k_query['filepath'].tolist()
                query_y += k_query['class_id'].tolist()

            yield support_X, support_y, query_X, query_y

    @staticmethod
    def image_pipeline(path_tensor: tf.Tensor) -> tf.Tensor:
        """Loads and decodes an image from a path string."""

        return tf.image.decode_image(tf.read_file(path_tensor), dtype=tf.float32)

    def tf_iterator(self, image_pipeline: Optional[Callable[[tf.Tensor], tf.Tensor]] = None) -> tf.data.Iterator:
        """Create a tensorflow iterator of batches of episodes from data.

        :param image_pipeline: a function mapping from a filename to an image data tensor.
        """
        if not image_pipeline:
            image_pipeline = self.image_pipeline

        def decode_image_tensor(t):
            """Load a list of images at once."""
            return tf.map_fn(image_pipeline, t, tf.float32)

        def prepare_outputs(x_s, y_s, x_q, y_q):
            return (
                (decode_image_tensor(x_s),
                 tf.one_hot(y_s, self.k_way),
                 decode_image_tensor(x_q)),
                tf.one_hot(y_q, self.k_way)
                )

        ds = tf.data.Dataset.from_generator(lambda: self,
                                            (tf.string, tf.int32, tf.string, tf.int32))

        ds = ds.map(prepare_outputs,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).make_one_shot_iterator()
