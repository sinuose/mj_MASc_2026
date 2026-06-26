import numpy as np


def getMesh(xmin, xmax, ymin, ymax, nx=50, ny=50):

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)

def getImageMesh(H, W, xmin = -0.5,xmax = 0.5, ymin = -0.5, ymax = 0.5):
    xv, yv = getMesh(xmin, xmax, ymin, ymax, nx=W, ny=H)
    return xv, yv

def cropCenter(img, H, W, pad=False):
    # get the center of the image
    xc = W//2
    yc = H//2

    tx_l = xc - W // 2
    tx_r = tx_l + W
    ty_l = yc - H // 2
    ty_r = ty_l + H

    if pad:
        # compute crop with safe clipping and pad if necessary
        pad_left = max(0, 0 - tx_l)
        pad_top = max(0, 0 - ty_l)
        pad_right = max(0, tx_r - W)
        pad_bottom = max(0, ty_r - H)

        # adjust crop coords because of padding
        tx_l += pad_left
        tx_r += pad_right
        ty_l += pad_top
        ty_r += pad_bottom

    return img[tx_l:tx_r, ty_l:ty_r]