import unittest
import simplejson as json
from racedetect import svm

class TestFeretPerson(unittest.TestCase):
    def setUp(self):
        self.path   = 'tests/data/00001'
        self.truths = json.load(open('tests/data/00001/truths.json'))
        self.person = svm.FeretPerson()

    def test_load(self):
        self.person.load(self.path)
        self.assertEqual(self.person.label, '00001')
        self.assertEqual(self.person.id, self.truths['id'])
        self.assertEqual(self.person.gender, self.truths['gender'])
        self.assertEqual(self.person.race, self.truths['race'])
        self.assertEqual(self.person.yob, self.truths['yob'])

class TestFeretImage(unittest.TestCase):
    def setUp(self):
        self.path   = 'tests/data/00001'
        self.person = svm.FeretPerson()
        self.person.load(self.path)

        self.image = self.person.images[0]

    def test_init(self):
        self.assertEqual(self.image.person_label, '00001')
        self.assertEqual(self.image.person_age, 50)
