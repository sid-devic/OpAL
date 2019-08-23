import os
import multiprocessing
import numpy as np


class ActiveLearningExperiment:
    def __init__(self, query_method, instance, gpu_index, dataset_name,
                 model, train_func, num_init, num_add_per_iter,
                 num_iter, x_train, y_train, x_test, y_test):
        """
        constructor

        :param query_method: Query method, child of Query class
        :param instance: Instance of ALexp for this configuration (we may need to run multiple)
        :param gpu_index: index of gpu to run the experiment on
        :param dataset_name: for logging purposes
        :param model: function to build a keras model (input_shape, num_classes)
        :param train_func: Training function (see example)
        :param num_init: number of initial labels
        :param num_add_per_iter: number of labels to add in each iteration
        :param num_iter: number of iterations
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        # https://stackoverflow.com/questions/26239695/parallel-execution-of-class-methods
        # If we would like to execute this class in parallel, we need to use
        # a process queue so that methods can pass data back up
        self.manager = multiprocessing.Manager()

        self.query_func = query_method
        self.query_name = self.query_func.__name__

        # build model
        self.model = model(x_train[0].shape, len(y_train[0]))
        self.train_func = train_func
        self.num_add_per_iter = num_add_per_iter
        self.num_iter = num_iter
        self.labeled_idx = np.random.choice(x_train.shape[0], num_init, replace=False)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_acc_hist = self.manager.list()
        self.labeled_idx_hist = self.manager.list()

        self.log_dir = 'logs/{0}_{1}_{2}_{3}_{4}.txt'.format(self.query_name, instance, num_init, num_add_per_iter, dataset_name)

    def _al_iter(self):
        """
        One step of active learning.
        1. train model
        2. log test acc and history of labeled indices
        3. add new labels to labeled set
        """
        # 1. Train using the given keras Model object
        test_acc, trained_model = self.train_func(self.model,
                                                  self.x_train[self.labeled_idx],
                                                  self.y_train[self.labeled_idx],
                                                  self.x_test,
                                                  self.y_test)
        # 2. Log
        self.test_acc_hist.append(test_acc)
        self.labeled_idx_hist.append(self.labeled_idx)
        with open(self.log_dir, 'w') as f:
            f.write(str(self.test_acc_hist).strip('[]'))
            f.write('\n')
            for _iter in range(len(self.labeled_idx_hist)):
                f.write(str(self.labeled_idx_hist[_iter]).strip('[]'))

        # 3. Query using the new model we just learned
        query_obj = self.query_func(trained_model, self.x_train.shape[0], self.y_train.shape[0], 1)
        self.labeled_idx = query_obj.query(self.x_train, self.y_train, self.labeled_idx, self.num_add_per_iter)

    def begin_al_loop(self, ):
        for _iter in range(self.num_iter):
            self._al_iter()
