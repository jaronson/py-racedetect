import os
import glob
import simplejson as json
import cv2
import utils
import config

from decorators import memoize
from detector import FaceDetector

class FeretImage(object):
    detector = FaceDetector()

    def __init__(self, person, truths):
        self.person_label = person.label

        self.__dict__.update(truths)
        self.__set_age(person)

    @memoize
    def normalized_mat(self):
        image = cv2.imread(self.path)

        try:
            rect = FeretImage.detector.find(image)[0]
        except IndexError:
            rect = None

        return utils.normalize_face(image, rect=rect)

    @memoize
    def histogram(self):
        pass

    def __set_age(self, person):
        capture_year    = self.capture_date.split('/')[-1]
        self.person_age = int(capture_year) - int(person.yob)

class FeretPerson(object):
    # takes a path to an asset directory
    # loading the dirname as the label,
    # truths.json and any images
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
