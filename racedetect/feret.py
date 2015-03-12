import os
import glob
import numpy as np
import simplejson as json
import cv2
import utils
import config

from skimage.feature import local_binary_pattern
from decorators import memoize
from detector import FaceDetector

class FeretImage(object):
    detector = FaceDetector()

    def __init__(self, person, truths):
        self.person_label = person.label
        self.face_found   = False

        self.__dict__.update(truths)
        self.__set_age(person)

    @memoize
    def mat(self):
        return cv2.imread(self.path)

    @memoize
    def normalized_mat(self):
        mat = self.mat()

        try:
            rect = FeretImage.detector.find(mat)[0]
            self.face_found = True
        except IndexError:
            rect = None

        return utils.normalize_face(mat, rect=rect)

    @memoize
    def histogram(self):
        return cv2.calcHist(
                [self.normalized_mat()],
                [0],
                None,
                [256],
                [0,256]
                )

    @memoize
    def lbp(self):
        image     = self.normalized_mat()
        neighbors = 8
        radius    = 1

        return local_binary_pattern(
                image,
                neighbors,
                radius,
                method='default'
                )

    @memoize
    def lbp_histograms(self):
        image    = self.normalized_mat()
        n_points = 8
        radius   = 1
        lbp      = local_binary_pattern(
                        image,
                        n_points,
                        radius,
                        method='default'
                        )

        # See: http://www.cse.unr.edu/~bebis/IJAIT12_Race.pdf
        # Their best case was 10x16 block size per 60x48
        # pixel image which equates to rows 1/6 high and
        # 1/3 wide.
        n_rows, n_cols   = (6, 3)
        image_h, image_w = lbp.shape[:2]
        block_w, block_h = (image_w / n_cols, image_h / n_rows)

        hists = []
        #blocks = np.empty((image_w, image_h), np.uint8)

        for i in range(n_rows):
            for j in range(n_cols):
                x = block_w * j
                y = block_h * i
                h = y + block_h
                w = x + block_w
                hist = cv2.calcHist(
                    [self.normalized_mat()],
                    [0],
                    None,
                    [256],
                    [0,256]
                    )
                hists.append(hist)
        return hists

    def __set_age(self, person):
        capture_year    = self.capture_date.split('/')[-1]
        self.person_age = int(capture_year) - int(person.yob)

class FeretPerson(object):
    def __init__(self, truths):
        self.image_data = truths.pop('images', None)
        self.__dict__.update(truths)
        self.__set_age()
        self.images = [ FeretImage(self, i) for i in self.image_data ]

    # Set age to date images taken - yob. The
    # year taken can vary from 1993 to 1996
    # so default to 1993. A better indication
    # can be had from each image.
    def __set_age(self):
        capture_year = 1993
        self.age     = capture_year - int(self.yob)

class FeretDatabase(object):
    RACE_KEY_MAP = {
            'Asian-Middle-Eastern':      'me',
            'Pacific-Islander':          'pi',
            'Native-American':           'na',
            'Asian-Southern':            'as',
            'Hispanic':                  'h',
            'Other':                     'o',
            'Asian':                     'a',
            'Black-or-African-American': 'b',
            'White':                     'w',
            }

    def __init__(self, path):
        self.path         = path
        self.race_key_map = FeretDatabase.RACE_KEY_MAP
        self.filepath     = os.path.join(path, 'manifest.json')
        self.manifest     = json.load(open(self.filepath))

    @memoize
    def all(self):
        return [ FeretPerson(p) for p in self.manifest ]

    @memoize
    def all_by_race(self):
        d = {}

        for p in self.all():
            key = self.race_key_map[p.race]

            if not d.has_key(key):
                d[key] = []

            d[key].append(p)
        return d
