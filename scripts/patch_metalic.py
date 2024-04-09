import numpy as np
import sys
import skimage as ski

path = sys.argv[1]
out_path = sys.argv[2]

coef = float(sys.argv[3])

image = ski.io.imread(path)


print(image[10,10])

image[:, :, 2:3] = (255 - image[:, :, 1:2]) * coef

print(image[10,10])

ski.io.imsave(out_path, image)