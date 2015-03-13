import unittest
import cv2
import numpy as np
from racedetect import algorithm

class TestWLD(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread('tests/data/wld.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    def test_stuff(self):
        mat = algorithm.webers_law_descriptors(self.image)
