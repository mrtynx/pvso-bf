import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def conv2d(image, kernel, padding=False):
    output = np.zeros_like(image)

    if padding:
        image = np.pad(image, ((1, 1), (1, 1)), mode="constant")

    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            output[i - 1, j - 1] = np.sum(kernel * image[i - 1: i + 2, j - 1: j + 2])

    return output


def sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gaussian_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    grad = np.zeros_like(image)
    angle = np.zeros_like(image)
    image = conv2d(image, gaussian_blur, padding=True)

    for i in range(1, grad.shape[0] - 1):
        for j in range(1, grad.shape[1] - 1):
            gx = np.sum(sobel_x * image[i - 1: i + 2, j - 1: j + 2])
            gy = np.sum(sobel_y * image[i - 1: i + 2, j - 1: j + 2])
            grad[i, j] = np.sqrt(gx ** 2 + gy ** 2)
            angle[i, j] = np.arctan2(gy, gx) * 180 / np.pi

    return grad, angle


def round_angle(image):
    output = np.zeros_like(image)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.round(np.mod(image[i, j], 180) / 45) * 45

    output[output == 180] = 0

    return output


def non_max_suppression(grad, angle):
    output = np.zeros_like(grad)

    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):

            q = 255
            r = 255
            ang = angle[i, j]
            if ang == 0:
                q = grad[i, j + 1]
                r = grad[i, j - 1]
            if ang == 45:
                q = grad[i + 1, j - 1]
                r = grad[i - 1, j + 1]
            if ang == 90:
                q = grad[i + 1, j]
                r = grad[i - 1, j]
            if ang == 135:
                q = grad[i - 1, j - 1]
                r = grad[i + 1, j + 1]

            if (grad[i, j] >= q) and (grad[i, j] >= r):
                output[i, j] = grad[i, j]
            else:
                output[i, j] = 0

    return output


def threshold(img, low=0.05, high=0.09):
    high = img.max() * high
    low = high * low

    img = np.array(img)
    output = np.zeros_like(img)

    output[img >= high] = 255
    output[(img >= low) & (img <= high)] = 25

    return output


if __name__ == "__main__":
    img = cv2.imread("img/building.jpg")
    img_gray = np.asarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (512, 512)))
    sobel = sobel(img_gray)
    rounded = round_angle(sobel[1])
    # plt.imshow(threshold(non_max_suppression(sobel[0], rounded)), cmap='gray')
    # plt.show()
    p = non_max_suppression(sobel[0], rounded)
    k = threshold(p)

    plt.imshow(k, cmap='gray')
    plt.show()
# %%
