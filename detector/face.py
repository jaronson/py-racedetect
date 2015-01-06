import numpy as np
import cv2

from collections import namedtuple

def create_cascades(**kwargs):
    files = [ cv2.CascadeClassifier("cascades/%s.xml" % fn) for fn in kwargs.values()]
    return namedtuple('CascadeSet', kwargs.keys())(*files)

class Face(object):
    cascades = create_cascades(face='haar/frontalface_default')

    def cascades(self):
        return self.__class__.cascades

class Eye(Face):
    cascades = create_cascades(face='haar/frontalface_default', eye='haar/eye')

    def detect(self, imagepath):
        c = cv2.createLBPHFaceRecognizer()
        return 0
        image = cv2.imread(imagepath)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascades.face.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(image ,(x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes      = self.cascades.eye.detectMultiScale(roi_gray)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
