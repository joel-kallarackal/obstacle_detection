# code
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data,morphology
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
import scipy.ndimage as nd
plt.rcParams["figure.figsize"] = (12,8)

# load images and convert grayscale
rocket = cv2.imread("images/road4.jpeg")
rocket_wh = rgb2gray(rocket)
 
# apply edge segmentation
# plot canny edge detection
edges = canny(rocket_wh)
plt.imshow(edges, interpolation='gaussian')
plt.title('Canny detector')
 
# fill regions to perform edge segmentation
fill_im = nd.binary_fill_holes(edges)
plt.imshow(fill_im)
plt.title('Region Filling')
 
# Region Segmentation
# First we print the elevation map
elevation_map = sobel(rocket_wh)
plt.imshow(elevation_map)
 
# Since, the contrast difference is not much. Anyways we will perform it
markers = np.zeros_like(rocket_wh)
markers[rocket_wh < 0.15] = 1 # 30/255
markers[rocket_wh > 0.5859375] = 2 # 150/255
 
plt.imshow(markers)
plt.title('markers')
 
# Perform watershed region segmentation
segmentation = watershed(image=elevation_map, markers=markers.astype(int))
 
plt.subplot(2,1,1)
plt.imshow(rocket)
plt.subplot(2,1,2)
plt.imshow(segmentation)
plt.show()
 
# # plot overlays and contour
# segmentation = nd.binary_fill_holes(segmentation - 1)
# label_rock, _ = nd.label(segmentation)
# # overlay image with different labels
# image_label_overlay = label2rgb(label_rock, image=rocket_wh)
 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
# ax1.imshow(rocket_wh)
# ax1.contour(segmentation, [0.8], linewidths=1.8, colors='w')
# ax2.imshow(image_label_overlay)
 
# fig.subplots_adjust(**margins)