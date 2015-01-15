from multiprocessing import Queue, Process, Pool
import log
import time
from matcher import FaceMatcher
from tracker import FaceTracker

logger = log.get_logger(__name__)

class QueueSet(object):
    def __init__(self, labels):
        for label in labels:
            setattr(self, label, Queue())

class Manager(object):
    def __init__(self, *queues):
        self.queues  = QueueSet(queues)
        self.workers = []

    def add(self, worker):
        worker.queues = self.queues
        self.workers.append(worker)

    def start(self):
        n = 0
        indices = []
        for worker in self.workers:
            indices.append(n)
            n += 1

        f = lambda n: self.workers[n].start()

        self.pool = Pool(processes=len(indices))
        self.pool.map_async(f, indices)

class BaseWorker(object):
    name = 'base_worker'
    app  = None

    def __init__(self, options = None):
        self.app = self.__class__.app(options)
        self.queues = None

    def start(self):
        proc = Process(name=self.__class__.name, target=self.app.run)
        proc.daemon = True
        logger.info('Starting %s daemon' % proc.name)
        proc.start()

class TrackWorker(BaseWorker):
    name = 'track_worker'
    app  = FaceTracker

class MatchWorker(BaseWorker):
    name = 'match_worker'
    app  = FaceMatcher
