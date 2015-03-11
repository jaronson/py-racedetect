import os
import glob
import simplejson as json
import utils
import config
from decorators import memoize

class FeretImage(object):
    def __init__(self, person, truths):
        self.person_label = person.label

        self.__dict__.update(truths)
        self.__set_age(person)

    def __set_age(self, person):
        capture_year    = self.capture_date.split('/')[-1]
        self.person_age = int(capture_year) - int(person.yob)

class FeretPerson(object):
    # takes a path to an asset directory
    # loading the dirname as the label,
    # truths.json and any images
    def __init__(self, path):
        self.path = path

    def load(self):
        parts      = self.path.split('/')
        self.label = parts[-1]

        self.__load_truths()

    def __load_truths(self):
        path   = os.path.join(self.path, 'truths.json')
        truths = json.load(open(path))
        images = truths.pop('images', None)

        self.__dict__.update(truths)
        self.__load_images(images)

    def __load_images(self, images):
        self.images = [ FeretImage(self, i) for i in images ]

class FeretDatabase(object):
    def __init__(self, path):
        self.path     = path
        filepath      = os.path.join(path, 'manifest.json')
        self.manifest = json.load(open(filepath))
        self.__all    = None

    @memoize
    def all(self):
        [ FeretPerson(self.path) for p in self.manifest ]
