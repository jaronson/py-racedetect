import unittest
from racedetect import config

class TestConfig(unittest.TestCase):

    def test_get(self):
        val = config.get('recognizer.face_training_limit')
        self.assertEqual(val, 50)

    def test_get_with_bad_key(self):
        val = config.get('recognizer.test_key')
        self.assertEqual(val, None)
