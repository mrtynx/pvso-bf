import cv2
import numpy as np

# Vykreslenie image pointov
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

sachovnica = cv2.imread("img/sachovnica.jpg")
sachovnica = cv2.resize(sachovnica, (240, 240))
gray = cv2.cvtColor(sachovnica, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
if ret:
    print("Found points on chessboard")
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    cv2.drawChessboardCorners(sachovnica, (7, 5), corners2, ret)
    cv2.imshow("Point detect", sachovnica)
    cv2.waitKey()

# Koeficienty kamery a kalibracna matica
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Kalibracia na inom obrazku
img = cv2.imread("img/sachovnica_2.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv2.imwrite("img/calibresult.png", dst)

# Re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

cv2.imshow("Original", img)
cv2.imshow("Undistort", dst)
cv2.waitKey()

