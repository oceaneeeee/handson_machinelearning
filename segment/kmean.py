import numpy as np
import cv2 as cv
import sys
K = int(sys.argv[1])
img = cv.imread(sys.argv[2])
cv.imshow('goc',img)
cv.waitKey(0)
img = cv.blur(img, (9, 9))
cv.imshow('nhieu',img)
cv.waitKey(0)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

label_image = label.reshape(img.shape[:2]).astype(np.uint8)
n_label, label_image = cv.connectedComponents(label_image)
mean = np.array([ cv.mean(img, mask = ((label_image == i) * 255).astype(np.uint8))[:3] for i in range(n_label) ]).astype(np.uint8)

print(f'{mean=}')

quant = mean[label_image.flatten()]
print(quant.shape)
quant = quant.reshape(img.shape)
cv.imshow('res2',quant)
cv.waitKey(0)
cv.destroyAllWindows()



# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()
