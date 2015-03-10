import unittest

from racedetect import face

class TestRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = face.Recognizer()

    def test_read_images(self):
        images, labels = self.recognizer.read_images(limit=10)
        unique_labels  = set(labels)

        self.assertEqual(len(set(labels)), 10)
