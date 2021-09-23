## import packages
import numpy as np

import matplotlib.pyplot as plt

## Load image
img = plt.imread("lenna.png")

# gray image generation
# img = np.mean(img, axis = 2, keepdims = True)

sz = img.shape

cmap = "gray" if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Ground Truth")
plt.show()

## 1-1. Inpainting: Uniform sampling
ds_y = 2
ds_x = 4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Sampling image")

## 1-2. Inpaining: random sampling
# # RGB
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# prob = 0.5
# msk = (rnd > prob).astype(np.float)

# Single Channel
rnd = np.random.rand(sz[0], sz[1], 1)
prob = 0.5
msk = (rnd > prob).astype(np.float)

msk = np.tile(msk, (1, 1, sz[2]))

dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Random mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Sampling image")

## 1-3. Inapainting: Gaussian sampling
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1

# gaus = a * np.exp(-((x-x0) ** 2 / ( 2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
# gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float)

gaus = a * np.exp(-((x-x0) ** 2 / ( 2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
rnd = np.random.rand(sz[0], sz[1], 1)
msk = (rnd < gaus).astype(np.float)
msk = np.tile(msk, (1, 1, sz[2]))

dst = img * msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Gaussia sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap = cmap, vmin = 0, vmax = 1)
plt.title("Sampling image")