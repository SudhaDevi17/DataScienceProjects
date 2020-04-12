import pickle

import numpy as np
import pandas as pd
from numpy import zeros
from sklearn.preprocessing import StandardScaler


def compute_cost(X, y, theta):
    '''
    Compute cost for linear regression
    '''

    m = y.size

    predictions = X.dot(theta)

    sq_errors = (predictions - y)

    current_cost_j = (1.0 / (2 * m)) * sq_errors.T.dot(sq_errors)

    return current_cost_j, np.mean(np.square(sq_errors))


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    gradient descent to learn theta
    '''
    m = y.size
    cost_past_val = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for n in range(theta_size):
            temp = X[:, n]
            temp.shape = (m, 1)
            errors_x1 = (predictions - y) * temp
            theta[n][0] = theta[n][0] - alpha * (1.0 / m) * errors_x1.sum()

        cost_past_val[i, 0], train_errors = compute_cost(X, y, theta)
    print('train error: ', train_errors)
    return theta, cost_past_val, train_errors


def train(path):
    # read data
    data = pd.read_csv(path, header=None)

    # process data
    data.columns = ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity',
                    'suction_side_displacement_thickness', 'scaled_sound_pressure']
    x = data[
        ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity', 'suction_side_displacement_thickness']]
    y = data['scaled_sound_pressure'].to_numpy()

    # Initializations
    scaler = StandardScaler()
    print(scaler.fit(x))
    X = scaler.transform(x)

    X = np.c_[np.ones(X.shape[0]), X]
    y = y.reshape(y.shape[0], 1)
    theta = zeros(shape=(X.shape[1], 1))

    alpha = 0.01  # Step size
    iterations = 1000  # No. of iterations
    np.random.seed(42)  # Set the seed

    thetas, cost, train_error = gradient_descent(X, y, theta, alpha, iterations)
    # save the model params and scaler object
    with open('model.pkl', 'wb') as f:
        pickle.dump(thetas, f)

    preprocessing_obj = open('scaler.obj', 'wb')
    pickle.dump(scaler, preprocessing_obj)

    return train_error


def predict(path):
    # load trained model
    # load save scaler object

    X_test = pd.read_csv(path)
    X_test.columns = ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity',
                      'suction_side_displacement_thickness', 'scaled_sound_pressure']
    X_test = X_test[
        ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity', 'suction_side_displacement_thickness']]

    try:
        file_pi2 = open('scaler.obj', 'rb')
        scaler = pickle.load(file_pi2)
        with open('model.pkl', 'rb') as f:
            slope = pickle.load(f)
        intercept = slope[0]
    finally:
        print("The model should be trained first ")

    return np.dot(scaler.inverse_transform(X_test), slope[1:]) + intercept
