import unittest
import face

image_path = '/Users/joshuaaronson/Documents/colorferet/output'

class TestRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = face.Recognizer()

    def test_read_images(self):
        images, labels = self.recognizer.read_images(image_path, limit=10)
        unique_labels  = set(labels)

        self.assertEqual(len(set(labels)), 10)

if __name__ == '__main__':
    unittest.main()
