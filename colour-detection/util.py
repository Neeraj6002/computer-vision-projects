import cv2
import numpy as np


def get_limits(color):

    c = np.uint8([[color]])

    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = int(hsvC[0][0][0])

    lowerLimit = np.array(
        [max(0, hue - 7), 150, 120],
        dtype=np.uint8
    )

    upperLimit = np.array(
        [min(179, hue + 7), 255, 255],
        dtype=np.uint8
    )

    return lowerLimit, upperLimit