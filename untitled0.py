import cv2
import numpy as np
import scipy as sci
from matplotlib import pyplot as plt

a = cv2.imread('grapes/grape2.jpeg', 0)

dft = cv2.dft(np.float32(a), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = dft
dft_shift = np.fft.fftshift(dft)

# Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
rows, cols = a.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 128
r_in = .1
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
# mask_area = np.logical_and(
#     np.logical_and(
#         (.05*(x - center[0] - 100) ** 2 + (y - center[1]-22) ** 2 >= r_in ** 2),
#         (.4*(x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2)),
#     (.05*(x - center[0] + 100) ** 2 + (y - center[1] + 22) ** 2 >= r_in ** 2)
# )

mask_area = np.logical_and(
        (-.1*(x - center[0]) ** 2 + 16*(y - center[1]) ** 2 >= r_in ** 2),
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2)
)

mask[mask_area] = 1

# dft_shift = dft_shift * mask
# _, dft_shift = cv2.threshold(dft_shift, -9999,99999,cv2.THRESH_BINARY)
# shape = dft_shift.shape[0:2]

dft_shift = np.divmod(dft_shift,64)*64
# div1 = np.asarray(np.divide(shape,10), np.uint16)
# w,h = div1
# w2, h2 = shape[:]-div1

# dft_shift[w:w2,h:h2,:] = 1
# # dft_shift[:w2,:h2,:] = 1

dft_magntude_spctrm = 20 * \
    np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

dft_unshiftd = np.fft.ifftshift(dft_shift)
idft = cv2.idft(dft_unshiftd)
idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

# fft = np.fft.fft(a)
# fft_shift = np.fft.fftshift(fft)
# fft_magntude_spctrm = 20 * np.log(np.abs(fft_shift))
# # fft_magntude_spctrm = np.asarray(fft_magntude_spctrm, dtype=np.uint8)
# fft_imback = np.fft.ifft(np.fft.ifftshift(fft_magntude_spctrm))
# fft_imback= np.asarray(fft_imback, np.float32)

# plt.imshow(a,cmap='gray')
fig, (ax1, ax2) = plt.subplots(2, 2)

ax1[0].imshow(a, cmap='gray')
ax2[0].imshow(dft_magntude_spctrm, cmap='gray')
ax2[1].imshow(idft, cmap='gray')
plt.show()


# fig, ax = plt.subplots(2, 2)

# ax[0,0].imshow(np.float32(a),cmap='gray')
# ax[0,1].imshow(idft, cmap='gray')
# ax[1,0].imshow(cv2.bitwise_xor(np.float32(a),idft), cmap='gray')
# plt.show()


plt.imshow(idft, cmap='gray')
plt.figure()
plt.imshow(cv2.bitwise_and(np.float32(a),idft), cmap='gray')
