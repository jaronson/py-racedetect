import config
from redis import StrictRedis

APP_CONFIG = config.APP_CONFIG

class Cache(object):

    def __init__(self):
        self.conn   = self.__connect()
        self.pubsub = self.conn.pubsub(ignore_subscribe_messages=True)

    def get_message(self):
        return self.pubsub.get_message()

    def publish(self, channel, data):
        return self.conn.publish(channel, data)

    def subscribe(self, *channels):
        return self.pubsub.subscribe(*channels)

    def __connect(self):
        return StrictRedis(host=APP_CONFIG['redis']['host'],
                port=APP_CONFIG['redis']['port'],
                db=0)
