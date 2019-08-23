from multiprocessing import Process

import keras
from sklearn.model_selection import train_test_split

from models.ResNet import ResNet18
from query_methods.RandomSampling import RandomSampling
from query_methods.CoreSet import CoreSetSampling
from query_methods.UncertaintySampling import UncertaintyEntropySampling
from al_exp import ActiveLearningExperiment


def train(model, x_train, y_train, x_test, y_test):
    """
    Train a model on the given dataset using simple ADAM opt. Take 20% for validation data.

    :param model: keras model
    :param x_train: train inputs
    :param y_train: train labels
    :param x_test: test inputs
    :param y_test: test labels
    :return: acc of model on test set, model itself
    """
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model.fit(x_train, y_train,
              batch_size=256,
              epochs=10,
              verbose=0,
              validation_data=(x_val, y_val))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score[1], model


def main():
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)
    print(train_x.shape, train_y.shape)
    num_gpus = 3
    methods = [CoreSetSampling, UncertaintyEntropySampling]
    processes = []

    for idx in range(min(len(methods), num_gpus)):
        experiment = ActiveLearningExperiment(query_method=methods[idx],
                                              instance=0,
                                              gpu_index=idx,
                                              dataset_name='cifar10',
                                              model=ResNet18,
                                              train_func=train,
                                              num_init=1000,
                                              num_add_per_iter=10,
                                              num_iter=10,
                                              x_train=train_x,
                                              y_train=train_y,
                                              x_test=test_x,
                                              y_test=test_y)
        processes.append(Process(target=experiment.begin_al_loop))
        
    # Start and join all processes (when complete)
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
