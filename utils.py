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
