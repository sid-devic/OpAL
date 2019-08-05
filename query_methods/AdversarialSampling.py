from .Query import QueryMethod, get_unlabeled_idx


class AdversarialSampling(QueryMethod):
    """
    An implementation of adversarial active learning, using cleverhans' implementation of DeepFool to generate
    adversarial examples.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled = X_train[unlabeled_idx]

        keras_wrapper = KerasModelWrapper(self.model)
        sess = K.get_session()
        deep_fool = DeepFool(keras_wrapper, sess=sess)
        deep_fool_params = {'over_shoot': 0.02,
                            'clip_min': 0.,
                            'clip_max': 1.,
                            'nb_candidate': Y_train.shape[1],
                            'max_iter': 10}
        true_predictions = np.argmax(self.model.predict(unlabeled, batch_size=256), axis=1)
        adversarial_predictions = np.copy(true_predictions)
        while np.sum(true_predictions != adversarial_predictions) < amount:
            adversarial_images = np.zeros(unlabeled.shape)
            for i in range(0, unlabeled.shape[0], 100):
                print("At {i} out of {n}".format(i=i, n=unlabeled.shape[0]))
                if i+100 > unlabeled.shape[0]:
                    adversarial_images[i:] = deep_fool.generate_np(unlabeled[i:], **deep_fool_params)
                else:
                    adversarial_images[i:i+100] = deep_fool.generate_np(unlabeled[i:i+100], **deep_fool_params)
            pertubations = adversarial_images - unlabeled
            norms = np.linalg.norm(np.reshape(pertubations,(unlabeled.shape[0],-1)), axis=1)
            adversarial_predictions = np.argmax(self.model.predict(adversarial_images, batch_size=256), axis=1)
            norms[true_predictions == adversarial_predictions] = np.inf
            deep_fool_params['max_iter'] *= 2

        selected_indices = np.argpartition(norms, amount)[:amount]

        del keras_wrapper
        del deep_fool
        gc.collect()

        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))