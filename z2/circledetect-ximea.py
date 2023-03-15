from ximea import xiapi
import cv2
import numpy as np

# %% Inicializacia kamery

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

# %% Detekcia kruhov RT na Ximea kamere

cv2.namedWindow("CircleDetect")


# Callbacky na trackbary
def on_param1_trackbar(val):
    pass


def on_param2_trackbar(val):
    pass


def on_min_radius_trackbar(val):
    pass


def on_max_radius_trackbar(val):
    pass


# Vytvorenie trackbarov
cv2.createTrackbar("Param1", "CircleDetect", 25, 100, on_param1_trackbar)
cv2.createTrackbar("Param2", "CircleDetect", 10, 100, on_param2_trackbar)
cv2.createTrackbar("MinRadius", "CircleDetect", 1, 100, on_min_radius_trackbar)
cv2.createTrackbar("MaxRadius", "CircleDetect", 1, 100, on_max_radius_trackbar)

while True:
    cam.get_image(img)
    src = cv2.resize(img.get_image_data_numpy(), (240, 240))

    # Houghova transformacia
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=cv2.getTrackbarPos("Param1", "CircleDetect"),
        param2=cv2.getTrackbarPos("Param2", "CircleDetect"),
        minRadius=cv2.getTrackbarPos("MinRadius", "CircleDetect"),
        maxRadius=cv2.getTrackbarPos("MaxRadius", "CircleDetect"),
        )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)

    cv2.imshow("CircleDetect", src)

# stop data acquisition
print("Stopping acquisition...")
cam.stop_acquisition()

# stop communication
cam.close_device()

print("Done.")
