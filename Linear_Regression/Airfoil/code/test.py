from unittest import TestCase

PATH = 'Linear_Regression/Airfoil/airfoil.csv'

#test
class TestAlgorithms(TestCase):
    from Linear_Regression.Airfoil.code.Airfoil import train
    train(PATH)
