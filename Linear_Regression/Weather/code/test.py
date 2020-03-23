from unittest import TestCase

PATH = 'Linear_Regression/Weather/weather.csv'


class TestAlgorithms(TestCase):
    from q4 import train
    train(PATH)
