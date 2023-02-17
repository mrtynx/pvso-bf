from ximea import xiapi
import cv2
import os
import numpy as np
img_path = 'img1/'
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(10000)
cam.set_param('imgdataformat', 'XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' % cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

i = 0
while cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (240, 240))
    cv2.imshow("test", image)
    cv2.waitKey(2000)
    cv2.imwrite(os.path.join(img_path, 'image' + str(i) + '.jpg'), image)
    i += 1
    if i == 4:
        break

# for i in range(3):
#     img_pth = 'img1/image' + str(i) + '.jpg'
#     img = cv2.imread(img_pth)
#     wname = 'img' + str(i)
#     cv2.imshow(wname, img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

# Filter uloha 1
img_pth = 'img1/image0.jpg'
img = cv2.imread(img_pth)
wname = 'img0'
wname1 = 'filter'
kernel = np.ones((3, 3), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)
cv2.imshow(wname, img)
cv2.imshow(wname1, dst)

cv2.waitKey()
cv2.destroyAllWindows()

# Otocenie o 90 stupnov
# img2 = cv2.imread('img1/image1.jpg')


# Zobrazenie iba R kanalu
wname2 = 'rkanal'
img3 = cv2.imread('img1/image2.jpg')
img3[:,:,0] = 0
img3[:,:,1] = 0
cv2.imshow(wname2, img3)

cv2.waitKey()
cv2.destroyAllWindows()

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')
