import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/road4.jpeg")
#img = cv2.GaussianBlur(img, (3,3), 0) 

hsvcolorspace = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lower_hsvcolorspace = np.array([0, 0, 0])
upper_hsvcolorspace = np.array([179, 255, 60])
mask = cv2.inRange(hsvcolorspace, lower_hsvcolorspace, upper_hsvcolorspace)

thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
  

plt.subplot(2,1,1)
plt.imshow(mask)
plt.subplot(2,1,2)
plt.imshow(thresh1)
plt.show()