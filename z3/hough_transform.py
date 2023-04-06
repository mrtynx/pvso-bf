import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from numba import jit

matplotlib.use("Qt5Agg")



def hough_intersect(img):
    thetas = np.deg2rad(np.arange(-90, 90))
    diag = int(np.round(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))
    rs = np.linspace(-diag, diag, 2 * diag)
    accumulator = np.zeros((2 * diag, len(thetas)))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                for k in range(len(thetas)):
                    r = j * np.cos(thetas[k]) + i * np.sin(thetas[k])
                    accumulator[int(r) + diag, k] += 1
    return accumulator, thetas, rs



def remove_close_duplicates(idxs, rel_tol):
    A = idxs[0][:]
    B = idxs[1][:]
    if len(A) != len(B):
        raise ValueError("Input arrays must have the same length")

    C, D = [], []
    seen_pairs = set()

    for i in range(len(A)):
        pair = (A[i], B[i])
        similar = False

        for seen_pair in seen_pairs:
            if math.isclose(seen_pair[0], A[i], rel_tol=rel_tol) and math.isclose(seen_pair[1], B[i], rel_tol=rel_tol):
                similar = True
                break

        if not similar and pair not in seen_pairs:
            seen_pairs.add(pair)
            C.append(A[i])
            D.append(B[i])

    return C, D


def polar_to_slope(rho, theta):
    angle = np.deg2rad(theta)
    a = -np.cos(angle) / np.sin(angle)
    b = rho / np.sin(angle)

    return a, b


def get_lines(rhos, thetas, space):
    if len(rhos) != len(thetas):
        raise ValueError("Input arrays must have the same length")

    lines = []
    x = np.linspace(0, space)
    for i in range(len(thetas)):
        a, b = polar_to_slope(rhos[i], thetas[i])
        lines.append(a * x + b)
    return lines


def plot_lines(img, rhos, thetas):
    if len(rhos) != len(thetas):
        raise ValueError("Input arrays must have the same length")

    x = np.linspace(0, img.shape[0])

    for i in range(len(thetas)):
        a, b = polar_to_slope(rhos[i], thetas[i])
        y = a * x + b
        plt.plot(x, y, color=[1, 0, 0])

    plt.imshow(img)
    plt.show()


def hough_lines(img, threshold, rel_tol=0.1):
    accumulator, thetas, rhos = hough_intersect(img)
    idxs_rhos, idxs_thetas = remove_close_duplicates(np.where(accumulator > threshold), rel_tol)
    rhos = rhos[idxs_rhos]
    thetas = np.rad2deg(thetas[idxs_thetas])

    return rhos, thetas
