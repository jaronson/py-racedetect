import unittest
from racedetect import svm

class TestTrainingAsset(unittest.TestCase):
    def setUp(self):
        self.asset = svm.TrainingAsset()

    def test_load(self):
        svm.load()
