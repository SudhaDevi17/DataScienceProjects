from unittest import TestCase

PATH = '../airfoil.csv'

#test
class TestAlgorithms(TestCase):
    from Airfoil import train
    train(PATH)
