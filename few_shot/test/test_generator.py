import numpy as np
import pandas as pd
import pytest
import uuid

from few_shot.dataset import FewShotEpisodeGenerator


N_CLASSES = 1000
N_SAMPLES = 100 * N_CLASSES

def create_df():
    filenames = [uuid.uuid4().hex for _ in range(N_SAMPLES)]
    classes = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

    return pd.DataFrame.from_dict({'filepath': filenames, 'class_name': classes})


@pytest.mark.parametrize('n', [1, 5])
@pytest.mark.parametrize('q', [5, 10])
@pytest.mark.parametrize('k', [5, 15, 50])
@pytest.mark.parametrize('df', [create_df()])
def test_generator(n, q, k, df):
    gen = FewShotEpisodeGenerator(df, 1000, n, k, q)
    it = iter(gen)

    for _ in range(50):
        support_x, support_y, query_x, query_y = next(it)

        # first make sure that we get the number of elements we expect
        assert len(support_x) == n * k
        assert len(support_y) == n * k
        assert len(query_x) == q * k
        assert len(query_y) == q * k

        # then make sure we get batch class ids (not the global class id), otherwise the one-hot encoding will break
        assert max(support_y) < k
        assert max(query_y) < k