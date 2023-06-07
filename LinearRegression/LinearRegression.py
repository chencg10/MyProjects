# Created by Chen Cohen Gershon
import numpy as np
import matplotlib.pyplot as plt


def Create_example_matrix(n, m):
    # 1.produce X matrix
    X = np.random.rand(n, m)
    return X


def produce_beta(m):
    # 2.produce beta vector
    beta = [i for i in range(1, m + 1)]
    return beta


def Create_noise_vector(m, sigma):
    # 3.produce epsilon vector with mean = 0 and sigma as variance
    epsilon = np.random.normal(0, sigma, m)

    return epsilon


def calc_Y(X, beta, epsilon):
    # 4.calc Y using equation : Y = X*beta + epsilon, @ is a vector mul operator
    Y = X @ beta + epsilon
    return Y


def calc_beta_est(X, Y):
    # 5.calc beta_est using equation : beta_est = ((X^T * X )^-1)(X^T * Y)
    beta_est = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ Y)
    return beta_est


def relative_error(beta, beta_est):
    R_error = np.abs(beta - beta_est) * 100 / np.abs(beta_est)
    print(R_error)


def main():
    n = 100 # Number of examples
    m = 8 # Number of features
    X = Create_example_matrix(n, m) # Create X matrix
    #----------sigma=1------------------------
    sigma = 1
    epsilon = Create_noise_vector(n, sigma)
    beta = produce_beta(m)
    Y1 = calc_Y(X, beta, epsilon)
    beta_est = calc_beta_est(X, Y1)
    relative_error(beta, beta_est)
    plt.plot(beta, beta_est)
    plt.xlabel("beta_est")
    plt.ylabel("beta")
    plt.title("Beta as function of Beta_est with sigma = 1")
    plt.show()
    # ----------sigma=1.5------------------------
    sigma = 1.5
    epsilon = Create_noise_vector(n, sigma)
    Y2 = calc_Y(X, beta, epsilon)
    beta_est = calc_beta_est(X, Y2)
    relative_error(beta, beta_est)
    plt.plot(beta, beta_est)
    plt.xlabel("beta_est")
    plt.ylabel("beta")
    plt.title("Beta as function of Beta_est with sigma = 1.5")
    plt.show()
    # ----------sigma=0.5------------------------
    sigma = 0.5
    epsilon = Create_noise_vector(n, sigma)
    Y3 = calc_Y(X, beta, epsilon)
    beta_est = calc_beta_est(X, Y3)
    relative_error(beta, beta_est)
    plt.plot(beta, beta_est)
    plt.xlabel("beta_est")
    plt.ylabel("beta")
    plt.title("Beta as function of Beta_est with sigma = 0.5")
    plt.show()


main()
