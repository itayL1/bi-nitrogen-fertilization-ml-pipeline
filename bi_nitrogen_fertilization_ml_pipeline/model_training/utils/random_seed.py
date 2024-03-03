import random

import numpy as np
import keras.utils


def set_random_seed_globally(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    keras.utils.set_random_seed(random_seed)
