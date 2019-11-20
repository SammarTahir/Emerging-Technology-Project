# Adapted from: https://docs.python.org/3/library/gzip.html
import gzip

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()

type(file_content)
file_content[0:4]

l = file_content[16:800]
type(l)

import numpy as np

image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8)

import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')

