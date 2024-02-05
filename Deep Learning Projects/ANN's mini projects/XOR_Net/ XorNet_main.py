import numpy as np
from XorNet_Class import XorNet
import matplotlib.pyplot as plt

# create the dataset:
X = np.zeros((4,2))
X[0,:] = [0,0]
X[1,:] = [0,1]
X[2,:] = [1,0]
X[3,:] = [1,1]

# create the labels:
y = np.zeros((4,1))
y[0] = -1
y[1] = 1
y[2] = 1
y[3] = -1




def train(X, y, epochs=100, lr=0.01):
    """
    This function trains the model
    :param lr: the learning rate for the gradient descent
    :param epochs: the number of epochs
    :param X: the input matrix
    :param y: the labels
    :return: the loss over the training process
    """
    loss_train = []
    model = XorNet()

    for epoch in (range(epochs)):
        loss_per_sample = 0
        for i in range(X.shape[0]):
            # forward pass:
            y_hat = model(X[i].reshape(2,1))
            # calc the loss:
            loss_per_sample += model.squred_loss(y[i], y_hat)
            # calc the gradients:
            gradients = model.gradient(X[i].reshape(2,1), y[i], y_hat)
            # update the parameters:
            for pidx, param in enumerate(model.get_parameters()):
                param -= lr * gradients[pidx]

        # save the loss of the epoch:
        loss_train.append(loss_per_sample)

    return model, loss_train


# train the model in a loop to get the best performance:
while True:
    model, loss = train(X, y, epochs=5000, lr=0.001)
    if loss[-1] < 0.2:
        # print prediction:
        print('Model prediction:')
        print(model(X[0].reshape(2,1))[0])
        print(model(X[1].reshape(2,1))[0])
        print(model(X[2].reshape(2,1))[0])
        print(model(X[3].reshape(2,1))[0])

        # print the parameters:
        # print('Model parameters:')
        # print(model.get_parameters())

        break
    else:
        print(f'Current Final Loss is: {loss[-1]: .5f}')


# plot the loss:
plt.plot(loss)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()
