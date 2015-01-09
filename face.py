import sys
import os
import glob
import re
import numpy as np
import cv2
import cv2.cv as cv
import utils
import xml.etree.ElementTree as ET

image_path = '/Users/joshuaaronson/Documents/yalefaces/gif'
model_path = 'tmp/facerec.xml'

class Face(object):
    count = 0

    def __init__(self, rect):
        self.rect        = rect
        self.available   = True
        self.training    = False
        self.delete      = False
        self.timer       = 40
        self.id          = Face.count
        self.match_label = None
        self.frames      = []

        Face.count += 1;

    def add_frame(self, frame):
        (x,y,w,h) = self.rect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        self.frames.append(gray[y:h, x:w])

    def dead(self):
        if self.timer < 0:
            return True
        return False

    def update(self, newRect):
        self.rect = newRect

    def decr(self):
        self.timer -= 1

class Detector(object):
    def __init__(self):
        self.cascade     = cv2.CascadeClassifier('cascades/haar/frontalface_alt.xml')
        self.detect_args = {
                'scaleFactor':  1.3,
                'minNeighbors': 4,
                'minSize':      (30,30),
                'flags':        cv.CV_HAAR_SCALE_IMAGE,
                }

    def find(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        rects = self.cascade.detectMultiScale(image, **self.detect_args)

        if len(rects) == 0:
            return []

        rects[:,2:] += rects[:,:2]

        return rects

class Recognizer(object):
    def __init__(self):
        self.model  = cv2.createLBPHFaceRecognizer()
        self.labels = None

    def load(self):
        self.model.load(model_path)

    def predict_from_frame(self, frame, rect):
        (x,y,w,h) = rect
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face = gray[y:h, x:w]

        return self.predict_from_image(face)

    def predict_from_image(self, image):
        label, confidence = self.model.predict(np.asarray(image))
        return [label, confidence]

    def save(self):
        self.model.save(model_path)

    def train(self, image_path):
        images, labels = self.__read_images(image_path)

        # Convert labels to 32bit integers. This is a workaround for 64bit machines.
        labels = np.asarray(labels, dtype=np.int32)
        self.model.train(np.asarray(images), labels)

    def update(self, images):
        # TODO: Fixme
        self.get_labels()
        label = self.next_label()
        self.labels.append(label)

        labels = len(images) * [label]

        self.model.update(np.asarray(images), np.asarray(labels))

    def next_label(self):
        return self.get_labels()[-1] + 1

    def get_labels(self):
        if self.labels is not None:
            return self.labels

        self.labels = []
        labels = ET.parse(model_path).find('labels').find('data').text
        labels = [ int(t) for t in labels.replace("\n",' ').split(' ') if not t == '' ]
        [self.labels.append(n) for n in labels if not self.labels.count(n)]

        return self.labels

    def __read_images(self, path, size=None):
        c = 0
        images = []
        labels = []
        files  = glob.glob('{0}/*'.format(path))

        for filename in files:
            try:
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                if size is not None:
                    image = cv2.resize(image, size)
                images.append(np.asarray(image, dtype=np.uint8))

                label = os.path.basename(filename).split('.')[0]
                label = int(re.search('\d+', label).group(0))
                labels.append(label)
            except IOError, (errno, strerror):
                print "I/O error({0}): {1}".format(errno, strerror)
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise

        return [images, labels]
