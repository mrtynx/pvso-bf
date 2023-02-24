from ximea import xiapi
import cv2
import os
import numpy as np

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
images = np.zeros([4, 240, 240, 4], dtype=np.uint8) # Obrazky sa budu ukladat tu, nie do suboru
while cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (240, 240))
    images[i, :, :, :] = image
    cv2.imshow("scanning...", image)
    cv2.waitKey(2000)
    i += 1
    if i == 4:
        break

# Vytvorenie mozaiky 
original_mosaic = np.vstack((np.hstack((images[0, :, :, :], images[1, :, :, :])),
                             np.hstack((images[2, :, :, :], images[-1, :, :, :]))))

# Filter - uloha 1
kernel = np.ones((3, 3), np.float32) / 25
dst = cv2.filter2D(images[0, :, :, :], -1, kernel)

# Otocenie o 90 stupnov - uloha 2
for i in range(images.shape[-1]):
    images[1, :, :, i] = np.rot90(images[1, :, :, i])

# Zobrazenie iba R kanalu - uloha 3
images[2, :, :, 0] = 0
images[2, :, :, 1] = 0

# Modifikovana mozaika
modif_mosaic = np.vstack((np.hstack((images[0, :, :, :], images[1, :, :, :])),
                          np.hstack((images[2, :, :, :], images[-1, :, :, :]))))

# Vykreslenie orig. a modif. mozaiky

cv2.imshow("Originalna mozaika", original_mosaic)
cv2.imshow("Modifikovana mozaika", modif_mosaic)

cv2.waitKey(100000)

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')
