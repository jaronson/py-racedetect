import cv2
import numpy as np
from skimage import feature

def histogram(image):
    return cv2.calcHist(
            [image],
            [0],
            None,
            [256],
            [0,256]
            )

def local_binary_pattern(image, neighbors=8, radius=1, method='default'):
    return feature.local_binary_pattern(
            image,
            neighbors,
            radius,
            method
            )

def lbp_histograms(image):
    lbp = local_binary_pattern(image)

    # See: http://www.cse.unr.edu/~bebis/IJAIT12_Race.pdf
    # Their best case was 10x16 block size per 60x48
    # pixel image which equates to rows 1/6 high and
    # 1/3 wide.
    n_rows, n_cols   = (6, 3)
    image_h, image_w = lbp.shape[:2]
    block_w, block_h = (image_w / n_cols, image_h / n_rows)

    hists = []

    for i in range(n_rows):
        for j in range(n_cols):
            x     = block_w * j
            y     = block_h * i
            h     = y + block_h
            w     = x + block_w
            block = np.asarray(lbp[y:h, x:w], dtype=np.uint8)
            hist  = cv2.calcHist(
                [block],
                [0],
                None,
                [256],
                [0,256]
                )
            hists.append(hist)
    return hists
