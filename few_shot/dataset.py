import os

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import tensorflow as tf

class OmniglotDataset(object):
    """Encapsulates the logic to build few-shot episodes from the Omniglot dataset."""

    def __init__(self, path, episodes_per_epoch, n_shot, k_way, q_queries, n_tasks=1):
        self.path = path
        self.episodes_per_epoch = episodes_per_epoch
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.n_tasks = n_tasks

        self.df = pd.DataFrame.from_records(self._scan_path(path))
        self.df['class_id'] = skp.LabelEncoder().fit_transform(self.df.class_name)

        self.n_classes = self.df.class_id.nunique()

    @staticmethod
    def _scan_path(path):
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

        return samples

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
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

    def tf_iterator(self):
        def decode_image_tensor(t):
            """Load a list of images at once."""
            return tf.map_fn(lambda x: tf.image.decode_image(tf.read_file(x), dtype=tf.float32), t, tf.float32)
        
        def prepare_outputs(x_s, y_s, x_q, y_q):
            return (
                (decode_image_tensor(x_s),
                tf.one_hot(y_s, self.n_classes),
                decode_image_tensor(x_q)),
                tf.one_hot(y_q, self.n_classes)
                )
        
        ds = tf.data.Dataset.from_generator(lambda: self,
                                            (tf.string, tf.int32, tf.string, tf.int32)) #,

        ds = ds.map(prepare_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds.make_one_shot_iterator()
        
