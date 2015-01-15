from redis import StrictRedis
import simplejson as json
import config

APP_CONFIG = config.APP_CONFIG

class Cache(object):
    def __init__(self):
        self.conn   = self.__connect()
        self.pubsub = self.conn.pubsub(ignore_subscribe_messages=True)

    def delete(self, key):
        return self.conn.delete(key)

    def get(self, key):
        return self.conn.get(key)

    def get_message(self):
        return self.pubsub.get_message()

    def keys(self, pattern):
        return self.conn.keys(pattern)

    def publish(self, channel, data):
        return self.conn.publish(channel, data)

    def set(self, key, data):
        return self.conn.set(key, data)

    def subscribe(self, *channels):
        return self.pubsub.subscribe(*channels)

    def __connect(self):
        return StrictRedis(host=APP_CONFIG['redis']['host'],
                port=APP_CONFIG['redis']['port'],
                db=0)
