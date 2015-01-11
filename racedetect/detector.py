import cv2
import cv2.cv as cv
import utils

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
        raise 'BaseDetector.find is not implemented'

    def load_cascade(self):
        raise 'BaseDetector.load_cascade is not implemented'

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

class Face(BaseDetector):
    def find(self, image):
        image = self.convert_image(image)
        return self.get_rects(image)

    def load_cascade(self):
        self.cascade = cv2.CascadeClassifier('cascades/lbp/frontalface.xml')

class Eye(BaseDetector):
    def load_cascade(self):
        self.cascade = cv2.CascadeClassifier('cascades/haar/eye.xml')

    def find(self, image):
        rects = self.get_rects(image)
        return rects
        #utils.draw_rects(image, rects, (0,255,0))
        #return image
