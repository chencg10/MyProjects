# Linear Regression with Noise

This code demonstrates linear regression with noise by generating example data, adding noise, and estimating the regression coefficients.

## Features

- Generates an example matrix `X` of size `(n, m)` where `n` is the number of examples and `m` is the number of features.
- Produces a beta vector `beta` of size `m` for the regression coefficients.
- Generates a noise vector `epsilon` of size `m` with mean 0 and specified variance.
- Calculates the response vector `Y` using the equation `Y = X * beta + epsilon`.
- Estimates the regression coefficients `beta_est` using the equation `beta_est = ((X^T * X)^-1)(X^T * Y)`.
- Calculates the relative error between `beta` and `beta_est`.
- Plots the relationship between `beta` and `beta_est` for different noise levels.

## Usage

1. Clone the repository:
