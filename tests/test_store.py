import unittest
import time
from racedetect import store

class TestStore(unittest.TestCase):

    def setUp(self):
        self.cache = store.Cache()

    def test_connection(self):
        self.assertTrue(self.cache.conn != None)

    def test_pubsub(self):
        self.cache.subscribe('test-channel1')
        self.cache.publish('test-channel1', 'test message')

        found  = None
        count  = 0
        mcount = 10

        while count < mcount:
            msg = self.cache.get_message()
            if msg is not None:
                found = msg['data']
                break
            count += 1
            time.sleep(0.05)

        self.assertEqual(found, 'test message')
