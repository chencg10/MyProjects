import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tqdm

data = None
train = None
test = None

# Get the data from mat file
def getdata(j):
    """
    Get the test and train data for a specific index j.
    :param j: Index of the data
    :return: test and train data for index j
    """
    global test
    global train

    test = data['test' + str(j)]
    train = data['train' + str(j)]
    return test, train


# Test the model on a new example
def test_model(X, w):
    """
    Test the trained model on a new example.
    :param X: Input data
    :param w: Weight vector
    :return: True if the prediction is >= 0.5, False otherwise
    """
    predict = sigmoid(X @ w)
    return predict >= 0.5


def cost_func(w, X, Y, N):
    """
    Calculate the cost function for logistic regression.
    :param w: Weight vector
    :param X: Input data matrix
    :param Y: Label vector
    :return: The cost function value
    """
    cost = []
    indx = range(1, 785)
    X = X[indx, :]
    Y = Y[indx]
    Yhat = sigmoid(X @ w)
    eps = 1e-3
    I = np.ones(Yhat.size)
    g = I - Yhat
    g[g < eps] = eps
    for i in Y:
        if i == 1:
            cost.append(Y * np.log(Yhat))
        else:
            cost.append(np.log(g))
    return np.squeeze(cost)


# This function returns a weights vector w
def train_process(X_train, X_test, y_train, y_test):
    """
    Train the logistic regression model.
    :param X_train: Training data matrix
    :param X_test: Test data matrix
    :param y_train: Training label vector
    :param y_test: Test label vector
    :return: The trained weight vector
    """
    N, D = X_train.shape
    batch = 64
    num_epoch = 100
    lr = 0.001
    epsilon = 1e-10
    # D - num of features, N - num of examples
    # Create w which is the weights vector, dim: 1xD
    # Start from random w:
    w = np.random.randn(D)
    acc_train = []
    acc_test = []
    loss_all = []

    for epoch_i in tqdm.tqdm(range(num_epoch)):
        acc_b = []
        for batch_i in range(0, len(X_train), batch):
            X = X_train[batch_i: batch_i + batch]
            y = y_train[batch_i: batch_i + batch]

            y_predict_soft = sigmoid(w @ X.T)
            y_predict_hard = np.round(y_predict_soft)

            acc_b.append(np.mean(y_predict_hard == y) * 100)

            loss = (1 / batch) * np.sum(np.inner(y, np.log(y_predict_soft + epsilon)) +
                                        np.inner((1 - y), np.log(1 - y_predict_soft + epsilon)))

            gradient_w = (1 / batch) * (X.T @ (y - y_predict_hard))

            w += lr * gradient_w

        y_predict_soft = sigmoid(w @ X_test.T)
        y_predict_hard = np.round(y_predict_soft)

        acc_test.append(np.mean(y_predict_hard == y_test) * 100)
        acc_train.append(np.mean(acc_b))
        loss_all.append(np.mean(loss))

    # Plotting the loss and accuracy
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss VS number of epochs')
    plt.plot(loss_all)
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS number of epochs')
    plt.plot(acc_test, label='test')
    plt.plot(acc_train, label='train')
    plt.legend()
    plt.show()

    return w


def sigmoid(z):
    """
    Calculate the sigmoid of z.
    :param z: Input value
    :return: The sigmoid value
    """
    return 1 / (1 + np.exp(-z))


def show_img_and_label(example, label):
    """
    Show the image and label of an example.
    :param example: Example data
    :param label: Example label
    """
    # Reshape the example into an image:
    example = numpy.reshape(example, [28, 28])
    plt1.imshow(example)
    plt1.show()
    print("The label of this example is: ", label + 1)


def main():
    # Get data from .mat file:
    global data
    data = scipy.io.loadmat('mnist_all.mat', squeeze_me=True)

    # We want to train the model only for the numbers 1 and 2.
    test1, train1 = getdata(1)
    test2, train2 = getdata(2)
    # Create data vector, X contains all data for the learning process
    X = np.concatenate([train1, test1, train2, test2])
    # Create labels vector Y, Y will contain all labels for data vector X
    label = np.concatenate([np.zeros(len(train1) + len(test1)), np.ones(len(train2) + len(test2))])

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=101)

    w = train_process(X_train, X_test, y_train, y_test)


main()
