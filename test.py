import keras
from models.LeNet import get_LeNet_model
from query_methods.RandomSampling import RandomSampling
from query_methods.CoreSet import CoreSetSampling
import numpy as np


def train(model, x_train, y_train, x_test, y_test):
    """
    Train a model on the given dataset using simple ADAM opt.

    :param model: keras model
    :param x_train: train inputs
    :param y_train: train labels
    :param x_test: test inputs
    :param y_test: test labels
    :return: acc of model on test set, model itself
    """
    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)
    print(x_train.shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=256,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score[1], model


def main():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)
    print(train_x.shape, train_y.shape)
    model = get_LeNet_model((28, 28, 1), labels=10
                            )

    input_shape = train_x.shape[0]
    num_labels = train_y.shape[0]
    print(input_shape, num_labels)

    query_method = CoreSetSampling(model, input_shape, num_labels, 1)
    labeled_idx = np.random.choice(train_x.shape[0], 1000, replace=False)
    acc, model = train(model, train_x[labeled_idx], train_y[labeled_idx], test_x, test_y)
    print(len(labeled_idx))

    # Query testing
    labeled_idx = query_method.query(np.expand_dims(train_x, 3), train_y, labeled_idx, 1000)
    print(len(labeled_idx))


if __name__ == '__main__':
    main()