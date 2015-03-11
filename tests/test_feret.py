import unittest
import simplejson as json
import numpy as np
from racedetect import config
from racedetect import feret

class TestFeretPerson(unittest.TestCase):
    def setUp(self):
        manifest    = json.load(open('tests/data/manifest.json'))
        self.truths = manifest[0]

    def test_load(self):
        person = feret.FeretPerson(self.truths)

        self.assertEqual(person.label, '00001')
        self.assertEqual(person.id, self.truths['id'])
        self.assertEqual(person.gender, self.truths['gender'])
        self.assertEqual(person.race, self.truths['race'])
        self.assertEqual(person.yob, self.truths['yob'])
        self.assertEqual(person.age, 50)

    def test_images(self):
        person = feret.FeretPerson(self.truths)
        image  = person.images[0]

        self.assertEqual(image.person_label, '00001')
        self.assertEqual(image.person_age, 50)
        self.assertEqual(image.filename, '00001_930831_fa_a.png')

class TestFeretImage(unittest.TestCase):
    def setUp(self):
        manifest    = json.load(open('tests/data/manifest.json'))
        self.truths = manifest[0]
        self.person = feret.FeretPerson(self.truths)
        self.image  = self.person.images[0]

    def test_normalized_mat(self):
        mat = self.image.normalized_mat()
        self.assertTrue(type(mat) == np.ndarray)
        self.assertTrue(len(mat) > 0)

class TestFeretDatabase(unittest.TestCase):
    def setUp(self):
        self.database     = feret.FeretDatabase(config.get('training.asset_path'))
        self.race_key_map = feret.FeretDatabase.RACE_KEY_MAP

    def test_all(self):
        self.database.all()

    def test_race_keys_uniqueness(self):
        keys = self.race_key_map.values()
        self.assertTrue(len(keys) == len(set(keys)))

    def test_all_by_race(self):
        by_race = self.database.all_by_race()
        keys    = self.race_key_map.values()

        for key in keys:
            self.assertTrue(len(by_race[key]) > 0)
