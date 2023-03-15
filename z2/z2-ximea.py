from ximea import xiapi
import cv2
import numpy as np
import glob

### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()


# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print("Opening first camera...")
cam.open_device()

# settings
cam.set_exposure(10000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
print("Exposure was set to %i us" % cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print("Starting data acquisition...")
cam.start_acquisition()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

sachovnica = cv2.imread("sachovnica.jpg")
sachovnica = cv2.resize(sachovnica, (240, 240))
gray = cv2.cvtColor(sachovnica, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
if ret:
    print("Nasiel")

# while True:
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     image = cv2.resize(image,(240,240))
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
#     if ret:
#         print("nasiel")
#         objpoints.append(objp)
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#         cv2.drawChessboardCorners(image, (7, 6), corners2, ret)

#
# cv2.imshow("test", image)
# cv2.waitKey(1000)

# stop data acquisition
print("Stopping acquisition...")
cam.stop_acquisition()

# stop communication
cam.close_device()

print("Done.")
