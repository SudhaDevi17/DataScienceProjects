from unittest import TestCase

PATH = 'Linear_Regression/Airfoil/airfoil.csv'

#test
class TestAlgorithms(TestCase):
    from  Airfoil import train
    train(PATH)
