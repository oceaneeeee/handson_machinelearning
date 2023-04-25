import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# Load in image, convert to gray scale, and Otsu's threshold
color = cv2.imread('example.jpg')
image = cv2.imread('example.jpg', 0)
th = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('th', th)
cv2.waitKey()
edges = cv2.Canny(image,100,200)
regions = 255 - edges
cv2.imshow('regions', regions)
cv2.waitKey()


# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(regions)
local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=regions)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=regions)

# Iterate through unique labels
total_area = 0
color_label = np.zeros(color.shape, dtype=np.uint8)
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(image.shape, dtype="uint8")
    mask[labels == label] = 255

    mean = cv2.mean(color, mask = mask)
    print(f'{mean=}')
    color_label[:,:] = [int(m) for m in mean[:-1]]
    color_label = cv2.bitwise_and(color_label, color_label, mask=mask)
    print(f'{color.dtype=}')
    print(f'{color_label.dtype=}')
    cv2.addWeighted(color_label, 1, color, 1, 0, color)

cv2.imshow('image', color)
cv2.waitKey()
