import time
import simplejson as json
import utils
import face
import log
import store

logger = log.get_logger(__name__)

class FaceMatcher(object):
    def __init__(self, opts = None):
        self.recognizer = face.Recognizer()
        self.store      = store.Cache()

        if opts and opts.train:
            logger.info('Training face recognizer')
            self.recognizer.train()
        else:
            logger.info('Loading face recognizer')
            self.recognizer.load()

    def __del__(self):
        logger.info('Saving face recognizer')
        self.recognizer.save()

    def run(self):
        self.store.subscribe('match')
        logger.info('Waiting for messages')

        while True:
            keys = self.store.keys('face/*')

            for key in keys:
                self.match(key)

            time.sleep(0.001)

    def match(self, key):
        ident       = key.split('/')[1]
        images      = [ utils.decode_image(i) for i in json.loads(self.store.get(key)) ]
        threshold   = face.Face.obj_distance_threshold
        label, dist = self.__predict(ident, images)

        if dist < threshold:
            logger.debug('Set match on face #{0} to label {1}'.format(ident, label))
        else:
            logger.debug('Training on face #{0}'.format(ident, label))
            label = self.recognizer.update(images)

        logger.debug('Added face #{0} to library, label: {1}'.format(ident, label))
        self.store.publish('track', 'face/{0}/{1}'.format(ident, label))
        self.store.delete(key)

        return (label, dist)

    # TODO: Account for number of matches in addition
    # to match distance?
    def __predict(self, ident, images):
        match_dict = {}
        matches    = []

        logger.info('Attempting to match face #{0}'.format(ident))

        for f in images:
            label, dist = self.recognizer.predict_from_image(f)

            if not label in match_dict.keys():
                match_dict[label] = [dist]
            else:
                match_dict[label].append(dist)

        for label in match_dict.keys():
            l = match_dict[label]
            matches.append((label, reduce(lambda x, y: x + y, l) / len(l)))

        top = sorted(matches, key=lambda t: t[1])[0]

        logger.debug('Found matches: {0}'.format(match_dict))
        logger.debug('Top match: {0}'.format(top))

        return top
