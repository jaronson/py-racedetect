import unittest
import simplejson as json
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

    def test_images(self):
        person = feret.FeretPerson(self.truths)
        image  = person.images[0]

        self.assertEqual(image.person_label, '00001')
        self.assertEqual(image.person_age, 50)
        self.assertEqual(image.filename, '00001_930831_fa_a.png')

class TestFeretDatabase(unittest.TestCase):
    def setUp(self):
        self.database = feret.FeretDatabase(config.get('training.asset_path'))

    def test_all(self):
        self.database.all()

    def test_all_by_race(self):
        by_race = self.database.all_by_race()
        keys    = [
                'Asian-Middle-Eastern',
                'Pacific-Islander',
                'Native-American',
                'Asian-Southern',
                'Hispanic',
                'Other',
                'Asian',
                'Black-or-African-American',
                'White'
                ]
        for key in keys:
            self.assertTrue(len(by_race[key]) > 0)
