from unittest import TestCase

PATH = '../weather.csv'


class TestAlgorithms(TestCase):
    from q4 import train
    train(PATH)
