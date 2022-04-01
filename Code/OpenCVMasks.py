import cv2
import numpy as np

img = cv2.imread("../PennFudanPed/PedMasks/PennPed00001_mask.png")
img = np.floor(img / np.max(img) * 255).astype(np.uint8)
cv2.imshow("Masked", img)
cv2.waitKey(5000)
