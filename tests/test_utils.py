import unittest
import cv2

from racedetect import utils
from racedetect.detector import FaceDetector

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()

    def test_equalize_halves(self):
        img = cv2.imread('tests/img/007/subject07-leftlight.jpg', cv2.COLOR_BGR2GRAY)
        cv2.imwrite('tmp/before-eq.jpg', img)

        rects = self.detector.find(img)

        for rect in rects:
            img = utils.crop_and_convert_face(img, rect)
            img = utils.equalize_halves(img)
            cv2.imwrite('tmp/after-eq.jpg', img)
