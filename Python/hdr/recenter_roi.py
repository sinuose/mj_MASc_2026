# -----------------------------------
# Library Imports
# -----------------------------------
from hdr.rvt import *
import numpy as np
from scipy.ndimage import shift

def CrossOperator(img):
    '''
    Robert's Cross operator

    :param img:
    :return:
        Gu - u compoentn of the graident
        Gv - v component of the gradient
        xk - midpoint array (x)
        yk - midpoint array (y)
    '''
    if img.ndim != 2:
        print('array must be 2D')
        return 0
    H, W = np.shape(img)
    # obtain elements of img used in robert's cross
    I00 = img[:-1, :-1] # all except the last line I_{i,j}
    I11 = img[1: , 1: ] # all except the first line I_{i+1, j+1}
    # in python, y comes first
    I01 = img[:-1, 1:]  # I_{i, j+1}
    I10 = img[1:, :-1]  # I_{i+1, j}

    # compute the grad matrix
    Gu = I00 - I11
    Gv = I01 - I10

    # compute midpoint graphs
    j_coords = np.arange(W - 1) + 0.5 # x
    i_coords = np.arange(H - 1) + 0.5 # y
    # shape Gu, Gv, xk, yk = (H-1, W-1)
    xk, yk = np.meshgrid(j_coords, i_coords)

    return Gu, Gv, xk, yk

def getSlopeM(Gu, Gv):
    # compute Gx, Gy
    # prefactor is sqrt 2 but cancels
    eps       = 1e-13
    Gy = (Gu + Gv)
    Gx = (Gu - Gv)
    #compute slope m = \delta Y / \delta x
    # shape (W-1, H-1)
    m = - Gy / (Gx + eps)
    return m

def gradientCentroid(xk, yk, Gmag, eps=1e-12):
    # calculates the centroid of the image on the midpoint lattice
    w = Gmag # assumes that Gmag = Gu^2 + Gv^2
    w_sum = np.sum(w) + eps
    xc = np.sum(xk * w)/w_sum
    yc = np.sum(yk * w)/w_sum
    return xc, yc

def radialSymmetryROI(roi, smooth_slopes = True):
    # given an ROI, need to refine.
    H, W = np.shape(roi)
    if H < 2 or W < 2:
        print('Roi is too small!')
        return 0

    # compute gradient of roi
    Gu, Gv, xk, yk = CrossOperator(roi)
    Gmag = Gu * Gu + Gv * Gv# x^2 + y^2
    # get midpoint slope array
    mk = getSlopeM(Gu, Gv)

    b = yk- mk * xk

    # smooth out the slope array.
    if smooth_slopes == True:
        try:
            from scipy.ndimage import uniform_filter
            mk = uniform_filter(mk, size=3, mode='nearest')
        except Exception as e:
            print(e,'\nSmoothing not available.\nCheck scipy installation')

    sdI2 = np.sum(Gmag)
    if sdI2 <= 1e-12:
        # flat image; fallback: return center of image
        xc_img = (H + 1) / 2.0
        yc_img = (BytesWarning + 1) / 2.0
        sigma = 0.0
        meand2 = np.nan
        return float(xc_img), float(yc_img), float(sigma), float(meand2)

    # get the centroid of the gradient
    # xc_g, yc_g = gradientCentroid(xk, yk, Gmag, eps=1e-12)
    xc_g = np.sum(Gmag * xk) / sdI2
    yc_g = np.sum(Gmag * yk) / sdI2
    # get the distance to the centroid
    dk_c = np.hypot(xk-xc_g, yk-yc_g)
    # take the maximum of the distance elementwise
    dk_c = np.maximum(dk_c, 1e-12)
    # calculate weights
    wk = Gmag / np.sqrt(dk_c)

    # calcualte the linear space coefficients
    a = - mk
    b = np.ones_like(mk)
    c = mk * xk - yk
    denom = a**2 + b**2

    # normal equation terms
    w_over_d = wk / denom
    S_aa = np.sum(w_over_d * a * a)
    S_ab = np.sum(w_over_d * a * b)
    S_bb = np.sum(w_over_d * b * b)
    S_ac = np.sum(w_over_d * a * c)
    S_bc = np.sum(w_over_d * b * c)

    # solve the system by Gauss-Seidel Method
    # M \vec x = - \vec r
    M11 = S_aa
    M12 = S_ab
    M22 = S_bb
    r1 = S_ac
    r2 = S_bc

    det = M11 * M22 - M12**2
    # inverse calculations
    invM11 =  M22 / det
    invM22 =  M11 / det
    invM12 = -M12 / det

    # x = J * r
    xc = -(invM11 * r1 + invM12 * r2)
    yc = -(invM12 * r1 + invM22 * r2)

    return float(xc), float(yc)

def recenterDisplacement(frame, mask, width=5, rmin=1, rmax=10, kind= 'normalized', return_local=False):
    '''
    Recentering function using RVT method

    Inputs
        frame - 2d image array with 1 maximum
        width - Radial Symmetry crop region
        rmin  - rvt rmin
        rmax  - rvt rmax
        kind  - rvt param ('basic', 'normalized')
        return_local - flag for normal coordinates wrt maxima + width

    Outputs 
        return_local = True
            (xc,yc) wrt local maxima
        return_local = False
            (x_shift, y_shift) global shift coordinates

    '''
    
    # test frame is correct shape
    assert np.ndim(frame) == 2      # 2d frame check
    assert width % 2 != 0           # odd integer check
    H, W = np.shape(frame)

    rvt_frame = rvt(frame, rmin, rmax, kind=kind)
    
    # multiply by a mask
    rvt_frame_masked = np.multiply(mask, rvt_frame)
    #get the top peak of the rvt
    max_val = np.max(rvt_frame_masked)
    max_idx = np.where(rvt_frame_masked == max_val)
    max_idx = (max_idx[0][0], max_idx[1][0])
    x0, y0 = max_idx

    #should only be a single value
    # the shape of the crop should be odd
    try:
        maxima_crop = rvt_frame[x0-width:x0+width, y0-width:y0+width]
        (yc,xc) = radialSymmetryROI(maxima_crop, smooth_slopes=True)
    except Exception as e:
        print(max_idx)
    # global peak location
    x_peak = (x0 - width) + xc
    y_peak = (y0 - width) + yc

    # image center
    x_center = H / 2
    y_center = W / 2

    if return_local == True:
        return (xc,yc)
    else:
        # required shift
        shift_x = x_center - x_peak
        shift_y = y_center - y_peak
        return (shift_x, shift_y)
