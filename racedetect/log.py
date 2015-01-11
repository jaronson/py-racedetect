import logging
import config

config = config.APP_CONFIG
levels = {
        'DEBUG': logging.DEBUG,
        'INFO':  logging.INFO,
        'WARN':  logging.WARN,
        'ERROR': logging.ERROR
        }

try:
    LEVEL = levels[config['log']['level']]
except KeyError:
    LEVEL = 'INFO'

logging.basicConfig(level=LEVEL)

def get_logger(name):
    return logging.getLogger(name)
