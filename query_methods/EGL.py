from .Query import QueryMethod, get_unlabeled_idx


class EGLSampling(QueryMethod):
    """
    An implementation of the EGL query strategy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def compute_egls(self, unlabeled, n_classes):

        # create a function for computing the gradient length:
        self.input_placeholder = K.placeholder(self.model.get_input_shape_at(0))
        self.output_placeholder = K.placeholder(self.model.get_output_shape_at(0))
        predict = self.model.call(self.input_placeholder)
        loss = K.mean(categorical_crossentropy(self.output_placeholder, predict))
        weights = [tensor for tensor in self.model.trainable_weights]
        gradient = self.model.optimizer.get_gradients(loss, weights)
        gradient_flat = [K.flatten(x) for x in gradient]
        gradient_flat = K.concatenate(gradient_flat)
        gradient_length = K.sum(K.square(gradient_flat))
        self.get_gradient_length = K.function([K.learning_phase(), self.input_placeholder, self.output_placeholder], [gradient_length])

        # calculate the expected gradient length of the unlabeled set (iteratively, to avoid memory issues):
        unlabeled_predictions = self.model.predict(unlabeled)
        egls = np.zeros(unlabeled.shape[0])
        for i in range(n_classes):
            calculated_so_far = 0
            while calculated_so_far < unlabeled_predictions.shape[0]:
                if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                    next = unlabeled_predictions.shape[0] - calculated_so_far
                else:
                    next = 100

                labels = np.zeros((next, n_classes))
                labels[:,i] = 1
                grads = self.get_gradient_length([0, unlabeled[calculated_so_far:calculated_so_far+next, :], labels])[0]
                grads *= unlabeled_predictions[calculated_so_far:calculated_so_far+next, i]
                egls[calculated_so_far:calculated_so_far+next] += grads

                calculated_so_far += next

        return egls

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        n_classes = Y_train.shape[1]

        # choose the samples with the highest expected gradient length:
        egls = self.compute_egls(X_train[unlabeled_idx], n_classes)
        selected_indices = np.argpartition(egls, -amount)[-amount:]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
