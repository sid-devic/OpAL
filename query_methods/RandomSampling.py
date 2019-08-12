from .Query import QueryMethod, get_unlabeled_idx
import numpy as np


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, x_train, y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(x_train, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))
