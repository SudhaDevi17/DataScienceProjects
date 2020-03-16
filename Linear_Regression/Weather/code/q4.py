import pickle

import numpy as np
import pandas as pd
from numpy import zeros


def data_processing(data, mode=None):
    # drop duplicates
    data.drop_duplicates(keep='first', inplace=True)
    # drop multicorrelated var
    columns_top_drop = ['Formatted_Date', 'Apparent_Temperature', 'Daily_Summary', 'Summary']
    # excluding these columns during selection
    X = data[['Precip_Type', 'Humidity', 'Win_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']]
    if mode == None:
        y = data['Temperature']  # .to_numpy()

    # impute missing values
    X['Precip_Type'] = X['Precip_Type'].fillna(X['Precip_Type'].value_counts().index[0])

    num_cols = ['Humidity', 'Win_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']
    # scaler = StandardScaler()
    # scaler.fit(X[num_cols])
    # X[num_cols] = scaler.transform(X[num_cols])
    with open('scaler.obj', 'rb') as f:
        scaler = pickle.load(f)

    X[num_cols] = scaler.transform(X[num_cols])

    # one hot encoded precip_type var
    X['Precip_Type'] = X['Precip_Type'].apply(lambda x: 1 if x == 'rain' else 0)

    if mode != None:
        return X
    return X, y


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
    data = pd.read_csv(path)

    # process data
    data.columns = ['Formatted_Date', 'Summary', 'Precip_Type', 'Temperature',
                    'Apparent_Temperature', 'Humidity', 'Win_Speed',
                    'Wind_Bearing', 'Visibility', 'Pressure',
                    'Daily_Summary']
    X, y = data_processing(data)

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

    return train_error


def predict(path):
    # load trained model
    # load save scaler object

    X_test = pd.read_csv(path)
    X_test.columns = ['Formatted_Date', 'Summary', 'Precip_Type', 'Temperature',
                      'Apparent_Temperature', 'Humidity', 'Win_Speed',
                      'Wind_Bearing', 'Visibility', 'Pressure',
                      'Daily_Summary']
    X_test = X_test[['Humidity', 'Win_Speed', 'Wind_Bearing', 'Visibility', 'Pressure', 'Precip_Type']]

    try:
        file_pi2 = open('scaler.obj', 'r')
        scaler = pickle.load(file_pi2)
        with open('model.pkl', 'rb') as f:
            slope = pickle.load(f)
        intercept = slope[0]
    finally:
        print("The model should be trained first ")
    X = data_processing(X_test, 1)
    X[['Humidity', 'Win_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']] = scaler.inverse_transform(
        X[['Humidity', 'Win_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']])
    return np.dot(X, slope[1:]) + intercept
