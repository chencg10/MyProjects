import numpy as np
# create a model class:
class XorNet:
    def __init__(self):
        """
        This function initializes the model parameters and the model itself
        :params: W, b1, b2, U
        """
        super().__init__()
        self.W = np.random.normal(0, 1, (2, 1))
        self.b2 = np.random.normal(0, 1)
        self.b1 = np.random.normal(0, 1, (2, 1))
        self.U = np.random.normal(0, 1, (2, 2))

    def forward(self, x):
        """
        This function calculates the model prediction for x
        :param x:
        :return: model prediction
        """
        # calc the model prediction for x:
        x = self.U.T @ x + self.b1
        # apply h: (relu activation)
        x = np.maximum(x, np.zeros(x.shape))
        # calc the model prediction for x:
        x = self.W.T @ x + self.b2
        return x

    def __call__(self, x):
        """
        This function is called when you call the object itself
        :param x:
        :return: calls the forward function
        """
        return self.forward(x)

    def get_parameters(self):
        """
        This function returns the model parameters
        :return: returns the model parameters
        """
        return [self.W, self.b1, self.b2, self.U]

    def gradient(self, x, y, yhat):
        """
        This function calculates the gradient of the model parameters
        :param x: the input matrix
        :param y: the labels
        :param yhat: the model prediction
        :return: a list of the gradients per parameter
        """
        # calc the gradient of W:
        return [self.derivative_W(x, y, yhat),
                self.derivative_b1(x, y, yhat),
                self.derivative_b2(x, y, yhat),
                self.derivative_U(x, y, yhat)
                ]

    def derivative_W(self, x, y, yhat):
        """
        This function calculates the gradient of W
        :param x: the input matrix
        :param y: the labels
        :param yhat: the model prediction
        :return: the gradient of W
        """
        return -2 * (y - yhat) * np.maximum(0, self.U.T @ x + self.b1)

    def derivative_b1(self, x, y, yhat):
        """
        This function calculates the gradient of b1
        :param x: the input matrix
        :param y: the labels
        :param yhat: the model prediction
        :return: the gradient of b1
        """
        d_h = np.where(self.U.T @ x + self.b1 >= 0, 1, 0)
        return -2 * self.W * (y - yhat) * d_h

    @staticmethod
    def derivative_b2(x, y, yhat):
        """
        This function calculates the gradient of b2
        :param x: the input matrix
        :param y: the labels
        :param yhat: the model prediction
        :return: the gradient of b2
        """
        return -2 * (y - yhat)

    def derivative_U(self, x, y, yhat):
        """
        This function calculates the gradient of U
        :param x:  the input matrix
        :param y:  the labels
        :param yhat:  the model prediction
        :return:  the gradient of U
        """
        d_h = np.where(self.U.T @ x + self.b1 >= 0, 1, 0)
        return -2 * (y - yhat) * self.W @ x.T * d_h

    @staticmethod
    def squred_loss(y, yhat):
        """
        This function calculates the loss
        :param y: the labels
        :param yhat: the model prediction
        :return: the loss
        """
        return np.sum((y - yhat) ** 2)



