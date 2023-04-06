import cv2
import matplotlib.pyplot as plt
import numpy as np

from canny import canny_edge_detect
from hough_transform import hough_lines, plot_lines

def on_param_1(val):
    pass

def on_param_2(val):
    pass

def on_param_3(val):
    pass

def on_param_4(val):
    pass



if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    cv2.namedWindow("LineDetect")

    cv2.createTrackbar("Canny threshold low", "LineDetect", 1, 100, on_param_1)
    cv2.createTrackbar("Canny threshold high", "LineDetect", 1,255, on_param_2)
    cv2.createTrackbar("Hough threshold", "LineDetect", 1,255, on_param_3)
    cv2.createTrackbar("Hough rel tolerance", "LineDetect", 0,1, on_param_4)

    while True:
        ret, src = vid.read()

        canny = canny_edge_detect(cv2.resize(src, (256, 256)),
                                  threshold_low=cv2.getTrackbarPos("Canny threshold low", "LineDetect"),
                                  threshold_high=cv2.getTrackbarPos("Canny threshold high", "LineDetect")
                                  )

        cv2.imshow("LineDetect", canny)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break