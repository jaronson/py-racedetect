import os
import glob
import simplejson as json
import utils
import config
import cv2
from decorators import memoize

class FeretImage(object):
    def __init__(self, person, truths):
        self.person_label = person.label

        self.__dict__.update(truths)
        self.__set_age(person)

    @memoize
    def get_mat(self):
        return cv2.imread(self.path)

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
        self.images = [ FeretImage(self, i) for i in self.image_data ]

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
        filepath          = os.path.join(path, 'manifest.json')
        self.manifest     = json.load(open(filepath))
        self.race_key_map = FeretDatabase.RACE_KEY_MAP
        self.__all        = None

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
