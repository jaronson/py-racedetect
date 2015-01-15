import cv2
import cv2.cv as cv
import utils
import config

class BaseDetector(object):
    def __init__(self):
        self.load_cascade()

        self.detect_args = {
                'scaleFactor':  1.3,
                'minNeighbors': 4,
                'minSize':      (30,30),
                'flags':        cv.CV_HAAR_SCALE_IMAGE,
                }

    def find(self, image):
        raise NotImplementedError

    def load_cascade(self):
        raise NotImplementedError

    def get_rects(self, image):
        rects = self.cascade.detectMultiScale(image, **self.detect_args)

        if len(rects) == 0:
            return []

        rects[:,2:] += rects[:,:2]

        return rects

    def convert_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray

class FaceDetector(BaseDetector):
    def find(self, image):
        image = self.convert_image(image)
        return self.get_rects(image)

    def load_cascade(self):
        self.cascade = cv2.CascadeClassifier(config.get('detector.face.cascade'))

class EyeDetector(BaseDetector):
    def load_cascade(self):
        self.cascade = cv2.CascadeClassifier(config.get('detector.eye.cascade'))

    def find(self, image):
        rects = self.get_rects(image)
        return rects
