import gc
from scipy.spatial import distance_matrix

from keras.models import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers
from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper

from models import *


def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, input_shape=(28,28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model
