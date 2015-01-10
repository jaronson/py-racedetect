import sys
import logging
import cv2
import utils
import face
import detector
import log

logger = log.get_logger(__name__)

class FaceTracker(object):
    def __init__(self, opts = None):
        self.options  = opts
        self.detector = detector.Face()
        self.matcher  = FaceMatcher(opts)
        self.faces    = []
        self.rects    = None

    def run(self):
        self.__init_video()

        while True:
            self.__read_video()
            self.rects = self.detector.find(self.frame_in)

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
                self.match_face(f)
                self.draw_face(f)

            cv2.imshow('img2', self.frame_out)

            if 0xFF & cv2.waitKey(5) == 27:
                break

        cv2.destroyAllWindows()

    def cull_faces(self):
        self.faces = [ f for f in self.faces if not f.delete ]

    def draw_face(self, face_obj):
        (x,y,w,h) = face_obj.rect
        color = (0, 255, 0)

        utils.draw_rects(self.frame_out, [ face_obj.rect ], color)
        utils.draw_msg(self.frame_out, (x, y), str(face_obj.id))

    def match_face(self, face_obj):
        self.matcher.match(face_obj, self.frame_in)

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
        self.frame_out = self.frame_in.copy()

class FaceMatcher(object):
    def __init__(self, opts = None):
        self.recognizer = face.Recognizer()

        if opts and opts.train:
            logger.info('Training face recognizer')
            self.recognizer.train()
        else:
            logger.info('Loading face recognizer')
            self.recognizer.load()

    def __del__(self):
        logger.info('Saving face recognizer')
        self.recognizer.save()

    def match(self, face_obj, frame):
        state = face_obj.get_state()

        if state == 'new':
            label, dist = self.__predict(face_obj, frame)
            logger.info('Untrained face detected, predicted label: {0}, distance: {1}'.format(label, dist))

            if dist >= 100:
                face_obj.add_frame(frame)
            else:
                logger.info('Setting match for face id#{0} to label {1}'.format(face_obj.id, label))
                face_obj.set_match(label)

        elif state == 'training':
            face_obj.add_frame(frame)

        elif state == 'unmatched':
            self.recognizer.update(face_obj.frames)
            label, dist = self.__predict(face_obj, frame)

            if dist < 100:
                logger.info('Matched face, predicted label: {0}, distance: {1}'.format(label, dist))
                face_obj.set_match(label)
            else:
                logger.info('No match found, predicted label: {0}, distance: {1}'.format(label, dist))
                face_obj.set_match(-1)

    def __predict(self, face_obj, frame):
        return self.recognizer.predict_from_frame(frame, face_obj.rect)
