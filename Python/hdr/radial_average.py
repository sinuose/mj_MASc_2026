# ----------------------------------------
# Radial Averaging
#
# the idea in this model is to do a polar integral and fit a function of f(r)
# centered on the minima of the function. This allows for a smooth fitting process
# and isolates the zpp parameters, as phi, and theta are integrated out.
# ----------------------------------------
import numpy as np

def radialAverage(roi, r_max=30, theta_range = (90,-90)):
    '''
    :param roi: 2D image array
    :param r_max:
    :param theta_range:
    :return r_ind: the emperical x values of the radial average
    :return radial_means: Radial averaging function f(r)
    :return mask: mask of where the radial average is applied
    '''
    # these are easily callable
    H, W = np.shape(roi)

    # pre-compute the constants for a given frame;

    cy = H // 2
    cx = W // 2

    # Coordinate grid
    y, x = np.indices((H,W))
    x_rel, y_rel = x - cx, y - cy
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.degrees(np.arctan2(y_rel, x_rel))

    # Restrict to wedge + radius
    # handle wrapping
    th1, th2 = theta_range
    if th1 > th2:  # e.g. (90, -90)
        mask_theta = (theta >= th1) | (theta <= th2)
    else:
        mask_theta = (theta >= th1) & (theta <= th2)

    mask = (r <= r_max) & mask_theta

    # compute integer radius map for entire image (same shape as `r`)
    r_int = np.floor(r).astype(int)   # or r.astype(int) — same for non-negative r

    # unique integer radii that actually appear inside the wedge
    unique_r = np.unique(r_int[mask])

    r_ind = unique_r  # these are integer radii

    # compute mean for each integer radius
    radial_means = np.empty(unique_r.size, dtype=float)
    for i, rr in enumerate(unique_r):
        sel = (mask) & (r_int == rr)
        radial_means[i] = roi[sel].mean()

    return r_ind, radial_means, mask


