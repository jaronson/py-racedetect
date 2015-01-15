import logging
import config

levels = {
        'DEBUG': logging.DEBUG,
        'INFO':  logging.INFO,
        'WARN':  logging.WARN,
        'ERROR': logging.ERROR
        }

LEVEL = config.get('log.level')
LEVEL = LEVEL if LEVEL else 'INFO'
LEVEL = levels[LEVEL]

logging.basicConfig(level=LEVEL)

def get_logger(name):
    return logging.getLogger(name)
