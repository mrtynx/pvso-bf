import cv2
import numpy as np

X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # X_kernel
Y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # Y_kernel

img = cv2.imread('img/lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (512, 512))


def convolve2d(kernel, img):

    kernel = np.array(kernel)
    img = np.array(img)
    res = np.zeros_like(img)
    worker = np.zeros_like(kernel)

    #
    for y in range(img.shape[1] - kernel.shape[0] + 1):
        for x in range(img.shape[0] - kernel.shape[0] + 1):

            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[0]):
                    worker[i, j] = kernel[i, j] * img[y + i, x + j]

            res[y + 1, x + 1] = worker.sum()

    return res


Gx = convolve2d(X, img_gray)
Gy = convolve2d(Y, img_gray)

# Gx = cv2.filter2D(img_gray, -1, np.array(X))
# Gy = cv2.filter2D(img_gray, -1, np.array(Y))
# G = np.sqrt(np.square(Gx) + np.square(Gy))

# G = G.astype(np.uint8)
#
# theta = np.arctan(Gy/Gx)

cv2.imshow("klfjsdisd", Gy)
cv2.waitKey(0)

