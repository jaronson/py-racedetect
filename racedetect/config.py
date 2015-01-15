import simplejson

APP_CONFIG = simplejson.load(open('config.json'))

def get(key):
    parts = key.split('.')
    d = APP_CONFIG

    try:
        for part in parts:
            d = d[part]
    except KeyError:
        return None

    return d
