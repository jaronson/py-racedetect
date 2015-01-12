import numpy as np
import cv2
import cv2.cv as cv
import color

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_msg(dest, (x, y), msg):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(dest, msg, (x + 1, y + 1), font, 1.0, color.BLACK, thickness=2, lineType=cv2.CV_AA) # Shadow
    cv2.putText(dest, msg, (x, y), font, 1.0, color.WHITE, lineType=cv2.CV_AA)

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def get_dist(arr_a, arr_b):
    return np.linalg.norm(np.asarray(arr_a) - np.asarray(arr_b))

def parse_and_convert_face(frame, rect):
    (x,y,w,h) = rect
    converted = frame.copy()
    converted = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    converted = cv2.equalizeHist(converted)
    converted = converted[y:h, x:w]
    return converted

def rotate_face(self, face_image):
    img   = face_image.copy()
    (h,w) = img.shape[:2]
    rects = Face.eye_detector.find(img)

    if len(rects) != 2:
        return

    (e1, e2) = rects

    if e1[0] < e2[0]:
        (r, l) = (e1, e2)
    else:
        (r, l) = (e2, e1)

    utils.draw_rects(img, rects, (0,255,0))

    direction = (r[0] - l[0], r[1] - l[1])
    rotation  = -math.atan2(float(direction[1]), float(direction[0]))
    mat       = cv2.getRotationMatrix2D((l[0], l[1]), rotation, 1.0)
    rotated   = cv2.warpAffine(img, mat, (w, h))
    return rotated

