from unittest import TestCase

PATH = '../airfoil.csv'


class TestAlgorithms(TestCase):
    from Airfoil import train
    train(PATH)
