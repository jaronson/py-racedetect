import numpy as np
import cv
import cv2

from collections import namedtuple

def create_cascades(**kwargs):
    files = [ cv2.CascadeClassifier("cascades/%s.xml" % fn) for fn in kwargs.values()]
    return namedtuple('CascadeSet', kwargs.keys())(*files)

class Face(object):
    def __init__(self):
        self.cascades = create_cascades(face='haar/frontalface_default')

    def detect(self, frame):
        image = np.asarray(frame[:,:])
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascades.face.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

        bitmap = cv.CreateImageHeader((image.shape[1], image.shape[0]), cv.IPL_DEPTH_8U, 3)
        cv.SetData(bitmap, image.tostring(), image.dtype.itemsize * 3 * image.shape[1])
        cv.ShowImage('w1', bitmap)
        cv.WaitKey(0)
        cv.DestroyAllWindows()
