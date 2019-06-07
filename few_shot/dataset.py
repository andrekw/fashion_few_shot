import os

import pandas as pd

class OmniglotDataset(object):
    """Encapsulates the logic to build few-shot episodes from the Omniglot dataset."""

    def __init__(self, path, episodes_per_batch):
        self.path = path
        self.episodes_per_batch = episodes_per_batch
        self.df = pd.DataFrame.from_records(self._scan_path(path))

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
