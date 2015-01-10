import sys
import os
import glob
import re
import math
import numpy as np
import cv2
import cv2.cv as cv
import utils
import detector
import xml.etree.ElementTree as ET

image_path = '/Users/joshuaaronson/Documents/colorferet/output'
model_path = 'tmp/facerec.xml'

class Face(object):
    obj_count       = 0  # Count of found faces for incrementing ids
    obj_timeout     = 60 # Timeout in seconds for each face
    obj_frame_count = 50 # The number of frames to collect for matching
    eye_detector    = detector.Eye()

    def __init__(self, rect):
        self.rect        = rect
        self.available   = True
        self.training    = False
        self.delete      = False
        self.id          = Face.obj_count
        self.timer       = Face.obj_timeout
        self.match_label = None
        self.frames      = []

        Face.obj_count += 1;

    def add_frame(self, frame):
        (x,y,w,h) = self.rect
        gray      = frame[y:h, x:w]
        gray      = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray      = cv2.equalizeHist(gray)
        #img       = self.rotate(gray)
        #cv2.imwrite(('tmp/frame-%s.jpg'% len(self.frames)), img)
        self.frames.append(gray)

    def rotate(self, face_image):
        img   = face_image.copy()
        (h,w) = img.shape[:2]
        rects = Face.eye_detector.find(img)

        if len(rects) != 2:
            return

        (e1, e2) = rects

        if e1[0] < e2[0]:
            (r, l) = (e1, e2)
        else:
            (r, l) = (e2, e1)

        utils.draw_rects(img, rects, (0,255,0))

        direction = (r[0] - l[0], r[1] - l[1])
        rotation  = -math.atan2(float(direction[1]), float(direction[0]))
        mat       = cv2.getRotationMatrix2D((l[0], l[1]), rotation, 1.0)
        rotated   = cv2.warpAffine(img, mat, (w, h))
        return rotated

    def dead(self):
        if self.timer < 0:
            return True
        return False

    def decr(self):
        self.timer -= 1

    def get_state(self):
        if self.match_label is None and len(self.frames) == 0:
            return 'new'
        if self.match_label is None and len(self.frames) < Face.obj_frame_count:
            return 'training'
        if self.match_label is None and len(self.frames) >= Face.obj_frame_count:
            return 'unmatched'
        if self.match_label is not None:
            return 'matched'
        if self.match_label == -1:
            return 'unknown'

    def set_match(self, label):
        self.match_label = label

    def update(self, newRect):
        self.rect = newRect

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

    def train(self):
        images, labels = self.read_images(image_path)

        # Convert labels to 32bit integers. This is a workaround for 64bit machines.
        labels = np.asarray(labels, dtype=np.int32)
        self.model.train(np.asarray(images), labels)
        self.labels = [ l for l in labels ]

    def update(self, images):
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

        # TODO: There should be a method of accessing the FaceRecognizer's
        # loaded label array. The C++ source doesn't seem to expose this
        # though, hence the xml parsing garbage.
        self.labels = []
        labels = ET.parse(model_path).find('labels').find('data').text
        labels = [ int(t) for t in labels.replace("\n",' ').split(' ') if not t == '' ]
        [self.labels.append(n) for n in labels if not self.labels.count(n)]

        return self.labels

    # Load images from a directory with the following format:
    # 001
    #   face_a.png
    #   face_b.png
    # 002
    #   face_a.png
    # The directory numbers are the labels.
    # In the above case, the labels will be [ 1, 1, 2 ].
    def read_images(self, path, limit=None, size=None, ext='png'):
        images  = []
        labels  = []
        subdirs = glob.glob('{0}/*'.format(path))
        count   = 0

        for subdir in subdirs:
            files = glob.glob('{0}/*.{1}'.format(subdir, ext))

            for f in files:
                image = self.__load_image(f, size=size)

                if image is not None:
                    images.append(image)
                    labels.append(int(os.path.basename(subdir)))

            count += 1

            if limit is not None and count >= limit:
                return [images, labels]

        return [images, labels]

    def __load_image(self, filepath, size=None):
        try:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if size is not None:
                image = cv2.resize(image, size)
            return np.asarray(image, dtype=np.uint8)
        except IOError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            return None
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
