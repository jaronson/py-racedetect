import unittest

from racedetect import config
from racedetect.detector import BaseDetector, FaceDetector

class TestDetector(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(NotImplementedError):
            BaseDetector()
