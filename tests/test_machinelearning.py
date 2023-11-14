from edvinspythonlib import machinelearning as ml
import unittest

class TestMachineLearning(unittest.TestCase):
    def test_multiplication(self):
        '''Tests multiplication function from machinelearning library'''
        assert ml.multiplication(0,394) == 0
        assert ml.multiplication(7,6) == 42
        assert ml.multiplication(2,3) != 7
        assert ml.multiplication(2,3) != 5
        assert round(ml.multiplication(2.3,3.4),2) == 7.82

    def test_addition(self):
        '''Tests addition function from machinelearning library'''
        assert ml.addition(0,394) == 394
        assert ml.addition(7,6) == 13
        assert ml.addition(2,3) != 6
        assert ml.addition(2,3) != 4
        assert round(ml.addition(2.3,3.4),1) == 5.7

if __name__ == '__main__':
    unittest.main()

