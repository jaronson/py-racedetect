import unittest
import simplejson as json
from racedetect import config
from racedetect import feret

class TestFeretPerson(unittest.TestCase):
    def setUp(self):
        self.path   = 'tests/data/00001'
        self.truths = json.load(open('tests/data/00001/truths.json'))
        self.person = feret.FeretPerson(self.path)

    def test_load(self):
        self.person.load()
        self.assertEqual(self.person.label, '00001')
        self.assertEqual(self.person.id, self.truths['id'])
        self.assertEqual(self.person.gender, self.truths['gender'])
        self.assertEqual(self.person.race, self.truths['race'])
        self.assertEqual(self.person.yob, self.truths['yob'])

class TestFeretImage(unittest.TestCase):
    def setUp(self):
        self.path   = 'tests/data/00001'
        self.person = feret.FeretPerson(self.path)
        self.person.load()

        self.image = self.person.images[0]

    def test_init(self):
        self.assertEqual(self.image.person_label, '00001')
        self.assertEqual(self.image.person_age, 50)
        self.assertEqual(self.image.filename, '00001_930831_fa_a.png')

class TestFeretDatabase(unittest.TestCase):
    def setUp(self):
        self.database = feret.FeretDatabase(config.get('training.asset_path'))

    def test_all(self):
        self.database.all(self.database)
