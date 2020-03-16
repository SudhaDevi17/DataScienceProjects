from unittest import TestCase

import numpy as np
import pandas as pd

TRAIN = 'Datasets/train.csv'
TEST = 'Datasets/test.csv'


class TestAlgorithms(TestCase):
    from DecisionTree import DecisionTree as dtree

    dtree_regressor = dtree()
    dtree_regressor.train(TRAIN)
    predictions = dtree_regressor.predict(TEST)
    raw_test_labels = pd.read_csv("Datasets/test_labels.csv", header=None)
    MSE = np.square(np.subtract(raw_test_labels[1], predictions)).mean()

    print('MSE: ', MSE)
