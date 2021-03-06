import cv2
import time
import utils
import face
import log
import store

from detector import FaceDetector

logger = log.get_logger(__name__)

class FaceTracker(object):
    def __init__(self, opts = None):
        self.options  = opts
        self.detector = FaceDetector()
        self.faces    = []
        self.rects    = None
        self.store    = store.Cache()

    def run(self):
        self.__init_video()
        self.store.subscribe('track')

        while True:
            self.__read_video()
            self.rects = self.detector.find(self.frame_in)

            msg   = self.store.get_message()
            ident = None
            label = None

            if msg is not None:
                ident, label = msg['data'].split('/')[1:3]

            # Scenario 1: face_list is empty
            if len(self.faces) == 0:
                # Make a face object for every rect
                self.faces = [ face.Face(r) for r in self.rects ]

            # Scenario 2: There are fewer face objects than rects
            elif len(self.faces) <= len(self.rects):
                self.match_new_rects()

            # Scenario 3: There are more face objects than rects
            else:
                self.mark_old_faces()

            self.cull_faces()

            for f in self.faces:
                if ident is not None:
                    if f.id == int(ident):
                        f.match_label = label

                self.match_face(f)
                self.draw_face(f)

            cv2.imshow('img2', self.frame_out)
            time.sleep(0.001)

            if 0xFF & cv2.waitKey(5) == 27:
                break

        cv2.destroyAllWindows()

    def cull_faces(self):
        self.faces = [ f for f in self.faces if not f.delete ]

    def draw_face(self, face_obj):
        (x,y,w,h) = face_obj.rect

        if face_obj.get_state() == 'matched':
            color = (0, 0, 255)
            msg   = "id: #{0}, match: {1}".format(face_obj.id, face_obj.match_label)
        else:
            color = (0, 255, 0)
            msg   = "id: #{0}".format(face_obj.id)

        utils.draw_rects(self.frame_out, [ face_obj.rect ], color)
        utils.draw_msg(self.frame_out, (x, y), msg)

    def match_face(self, face_obj):
        state = face_obj.get_state()

        if state == 'new':
            logger.info('Detected new face #{0}'.format(face_obj.id))
            face_obj.add_frame(self.frame_in)
        elif state == 'training':
            logger.debug('Adding frame for face #{0}, #{1}'.format(face_obj.id, face_obj.frame_count))
            face_obj.add_frame(self.frame_in)
        elif state == 'unmatched':
            logger.debug('Publishing frames for face #{0}'.format(face_obj.id))
            face_obj.publish_frames()

    def mark_old_faces(self):
        # All face objects start out as available
        for f in self.faces:
            f.available = True

        for r in self.rects:
            record = 50000
            index  = None
            i      = 0

            for f in self.faces:
                dist = self.__get_rect_dist(f.rect, r)
                if dist < record:
                    record = dist
                    index  = i
                i += 1

            match = self.faces[index]
            match.available = False
            match.update(r)

        for f in self.faces:
            if f.available:
                f.decr()

                if f.dead():
                    f.delete = True

    def match_new_rects(self):
        used = len(self.rects) * [False]

        for f in self.faces:
            record = 50000
            index  = None
            i      = 0

            for r in self.rects:
                dist = self.__get_rect_dist(f.rect, r)

                if dist < record and not used[i]:
                    record = dist
                    index  = i
                i += 1

            used[index] = True
            f.update(self.rects[index])

        i = 0
        for r in self.rects:
            if not used[i]:
                self.faces.append(face.Face(r))
            i += 1

    def __get_rect_dist(self, r1, r2):
        return utils.get_dist((r1[0], r1[1]), (r2[0], r2[1]))

    def __init_video(self):
        self.video     = None
        self.frame_in  = None
        self.frame_out = None

        self.video = cv2.VideoCapture(0)

    def __read_video(self):
        ret, self.frame_in = self.video.read()
        self.frame_out     = self.frame_in.copy()


