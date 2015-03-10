import sys
import os
import glob
import re
import math
import numpy as np
import cv2
import cv2.cv as cv
import simplejson as json
import xml.etree.ElementTree as ET

import config
import utils
import detector
import log
import store

logger = log.get_logger(__name__)

class Face(object):
    # Count of found faces for incrementing ids
    obj_count = 0

    # Timeout in seconds for each face
    obj_timeout = config.get('matcher.face_timeout_seconds')

    # The number of frames to collect for matching
    obj_frame_count = config.get('matcher.frame_collect_count')

    # The number below which a match is considered found
    obj_distance_threshold = config.get('matcher.distance_threshold')

    def __init__(self, rect):
        self.rect        = rect
        self.available   = True
        self.delete      = False
        self.id          = Face.obj_count
        self.timer       = Face.obj_timeout
        self.match_label = None
        self.frames      = []
        self.store       = store.Cache()
        self.published   = False
        self.frame_count = 0

        self.store.subscribe('match')

        Face.obj_count += 1;

    def add_frame(self, frame):
        converted = utils.crop_and_convert_face(frame, self.rect)
        self.frames.append(converted)
        self.frame_count += 1

    def dead(self):
        if self.timer < 0:
            return True
        return False

    def decr(self):
        self.timer -= 1

    def get_state(self):
        if self.match_label is None and self.frame_count == 0:
            return 'new'
        if self.match_label is None and self.frame_count < Face.obj_frame_count:
            return 'training'
        if self.match_label is None and self.frame_count >= Face.obj_frame_count:
            if self.published is True:
                return 'published'
            else:
                return 'unmatched'
        if self.match_label is not None:
            return 'matched'
        if self.match_label == -1:
            return 'unknown'

    def publish_frames(self):
        if self.published is True:
            return

        key  = ('face/%s' % self.id)
        data = [ utils.encode_image(f) for f in self.frames ]
        ret  = self.store.set(key, json.dumps(data))
        self.published = True

    def set_match(self, label):
        self.match_label = label

    def update(self, newRect):
        self.rect = newRect

class Recognizer(object):
    def __init__(self):
        self.model      = cv2.createLBPHFaceRecognizer()
        self.model_path = None
        self.labels     = None

    def load(self, path=None):
        path = path if path else config.get('recognizer.model_path')
        self.model_path = path

        if os.path.isfile(path):
            return self.model.load(path)

        return False

    def predict_from_frame(self, frame, rect):
        converted = utils.crop_and_convert_face(frame, rect)
        return self.predict_from_image(converted)

    def predict_from_image(self, image):
        label, confidence = self.model.predict(np.asarray(image))
        return (label, confidence)

    def save(self, outpath=None):
        outpath = outpath if outpath else config.get('recognizer.model_path')
        self.model.save(outpath)

    def train(self, **kwargs):
        images, labels = self.read_images(**kwargs)

        # Convert labels to 32bit integers. This is a workaround for 64bit machines.
        labels = np.asarray(labels, dtype=np.int32)

        logger.debug('Training model on {0} images'.format(len(images)))
        self.model.train(np.asarray(images), labels)
        self.labels = [ l for l in labels ]

    def update(self, images):
        self.get_labels()
        label = self.next_label()
        self.labels.append(label)

        labels = len(images) * [label]

        self.model.update(np.asarray(images), np.asarray(labels))
        return label

    def next_label(self):
        return self.get_labels()[-1] + 1

    def get_labels(self):
        if self.labels is not None:
            return self.labels

        # TODO: There should be a method of accessing the FaceRecognizer's
        # loaded label array. The C++ source doesn't seem to expose this
        # though, hence the xml parsing garbage.
        self.labels = []
        labels = ET.parse(self.model_path).find('labels').find('data').text
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
    def read_images(self, path=None, limit=None, size=None, ext=None):
        # Set some defaults
        path  = path if path else config.get('recognizer.image_path')
        limit = int(limit) if limit else config.get('recognizer.face_training_limit')

        if ext is None:
            ext = config.get('recognizer.image_extension') or 'png'

        subdirs = glob.glob('{0}/*'.format(path))
        images  = []
        labels  = []
        count   = 0

        for subdir in subdirs:
            files = glob.glob('{0}/*.{1}'.format(subdir, ext))

            for f in files:
                logger.debug('Loading file: {0}'.format(f))
                image = self.__load_image(f, size=size)

                if image is not None:
                    images.append(image)
                    labels.append(int(os.path.basename(subdir)))

            count += 1

            if limit is not None and count >= limit:
                break

        return [images, labels]

    def __load_image(self, filepath, size=None):
        try:
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if size is not None:
                image = cv2.resize(image, (size, size))

            return np.asarray(image, dtype=np.uint8)
        except IOError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            return None
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
