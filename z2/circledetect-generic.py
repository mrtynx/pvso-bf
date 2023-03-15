import cv2
import numpy as np

# %% Detekcia kruhov RT na generickej kamere

vid = cv2.VideoCapture(0)

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
    ret, src = vid.read()

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
