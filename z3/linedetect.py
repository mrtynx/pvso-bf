import cv2
import matplotlib.pyplot as plt
import numpy as np

from z3.canny import canny_edge_detect
from z3.hough_transform import hough_lines, plot_lines, get_lines


if __name__ == "__main__":
    img = np.asarray(cv2.resize(cv2.imread("img/building.jpg"), (256, 256)))
    canny = canny_edge_detect(img, threshold_low=60, threshold_high=170, hysteresis=False)

    lines = hough_lines(canny, threshold=80, rel_tol=0.09)

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].imshow(img)
    axes[0].set_title("Original picture")

    axes[1].imshow(canny, cmap='gray')
    axes[1].set_title("Canny edge detect")

    plt.subplots_adjust(wspace=0.4)
    plt.show()

    plot_lines(img, *lines)

