#!/usr/bin/python3
import numpy as np
import cv2
import sys
from functools import reduce

class Segmenter:
    def __init__(self, colors, pepper_size):
        self.colors = colors
        self.pepper_size = pepper_size

    def __call__(self, image):
        ret = self.colosest_color(image)
        #ret = self.remove_pepper(ret)
        ret = self.index_to_color(ret)
        return ret

    def colosest_color(self, image):
        image = cv2.blur(image, (self.pepper_size, self.pepper_size))
        height = image.shape[0]
        width = image.shape[1]
        segment = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(height):
            print(f'#', end='', flush=True)
            for j in range(width):
                pixel = image[i, j]
                dists = [np.linalg.norm(pixel - x) for x in self.colors]
                min_idx = np.argmin(dists)
                segment[i, j] = min_idx
        return segment
    
    def remove_pepper(self, idx_image):
        retval, labels = cv2.connectedComponents(idx_image)
        label_idx = [0] * retval
        kernel = np.ones((3,3),np.uint8)
        height = idx_image.shape[0]
        width = idx_image.shape[1]
        for i in range(retval):
            zone = np.where(labels == i)
            if len(zone[0]) > self.pepper_size: continue

            count = {}
            for coord in list(zip(zone[0], zone[1])):
                for j in range(coord[0] - 1, coord[0] + 2):
                    if j < 0: continue
                    if j >= height: continue
                    for k in range(coord[1] - 1, coord[1] + 2):
                        if k < 0: continue
                        if k >= width: continue
                        if labels[j, k] != labels[coord]:
                            if labels[j, k] not in count:
                                count[labels[j, k]] = 1
                            else:
                                count[labels[j, k]] += 1

            fusion_to_label = max( count, key=lambda item: count[item])
            fusion_to_zone = np.where(labels == fusion_to_label)
            fusion_to_idx = idx_image[fusion_to_zone[0][0], fusion_to_zone[1][0]]
            print(f'{i} -> {fusion_to_idx}', end=', ', flush=True)
            for coord in list(zip(zone[0], zone[1])):
                idx_image[coord] = fusion_to_idx

        return idx_image

    def index_to_color(self, idx_image):
        height = idx_image.shape[0]
        width = idx_image.shape[1]
        image = np.empty([height, width, 3])
        for i in range(height):
            for j in range(width):
                image[i, j] = self.colors[idx_image[i, j]]
        return image





if __name__ == "__main__":
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    print(f'{image.shape=}')
    n_color = 3
    div = 255/(n_color - 1)
    color_elem = []
    for i in range(n_color):
        color_elem.append(int(div*i))
    colors = []
    for i in color_elem:
        for j in color_elem:
            for k in color_elem:
                colors.append([i, j, k])
    print(f'{colors=}')
    #colors = [
    #        [0, 0, 0], 
    #        [0, 0, 255], 
    #        [0, 255, 0], 
    #        [0, 255, 255], 
    #        [255, 0, 0], 
    #        [255, 0, 255], 
    #        [255, 255, 0], 
    #        [255, 255, 255], 
    #        ]

    pepper_size = 3
    segment = Segmenter(colors, pepper_size = pepper_size)
    ret = segment(image)
    cv2.imwrite(image_path + f'.{pepper_size}.ret.jpg', ret)


