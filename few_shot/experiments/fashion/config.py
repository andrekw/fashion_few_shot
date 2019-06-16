SHOTS = [5, 1]
TEST_K_WAY = [15, 5]

EPS_PER_EPOCH = 100
N_EPOCHS = 100
TEST_EPS = 1000
PATIENCE = 1

N_QUERIES_TRAIN = 5
N_QUERIES_TEST = 5
K_WAY_TRAIN = 30

DATASET_PATH = 'datasets/fashion-dataset'
MIN_ROWS = N_QUERIES_TRAIN + max(SHOTS)
N_VAL_CLASSES = 16
RESULTS_FOLDER = 'results'

IMG_SHAPE = (160, 120, 3)

lr = 1e-3