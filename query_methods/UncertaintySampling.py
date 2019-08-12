import numpy as np

from .Query import QueryMethod, get_unlabeled_idx


class UncertaintyEntropySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, x_train, y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(x_train, labeled_idx)
        predictions = self.model.predict(x_train[unlabeled_idx, :])

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 20

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t, :, :] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, x_train, y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(x_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        uncertainties = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        split = 128  # split into iterations of 128 due to memory constraints
        while i < unlabeled_idx.shape[0]:

            if i+split > unlabeled_idx.shape[0]:
                preds, unc = self.dropout_predict(x_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
                uncertainties[i:] = unc
            else:
                preds, unc = self.dropout_predict(x_train[unlabeled_idx[i:i + split], :])
                predictions[i:i+split] = preds
                uncertainties[i:i+split] = unc
            i += split

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintyEntropySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using maximal entropy as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 100

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t,:,:] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, x_train, y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(x_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        while i < unlabeled_idx.shape[0]: # split into iterations of 1000 due to memory constraints

            if i+1000 > unlabeled_idx.shape[0]:
                preds, _ = self.dropout_predict(x_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
            else:
                preds, _ = self.dropout_predict(x_train[unlabeled_idx[i:i + 1000], :])
                predictions[i:i+1000] = preds

            i += 1000

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
