# Author; Martin Jasinski
# Radial Symmetry Localization Algorithmn based on Partharasy's refinement algorithmn.

import numpy as np


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
        xc_img = (Nx + 1) / 2.0
        yc_img = (Ny + 1) / 2.0
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


def visualize_radial_symmetry(roi , xc, yc, xc_g, yc_g, Gu, Gv, xk, yk, mk):
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    # ---------- AX[0]: ROI, slopes, and centroids ----------
    ax[0].imshow(roi, cmap='gray', origin='upper')
    ax[0].quiver(xk, yk, np.cos(np.arctan(mk)), np.sin(np.arctan(mk)),
                 color='cyan', angles='xy', scale_units='xy', scale=10, alpha=0.5)
    ax[0].scatter(xc_g, yc_g, color='yellow', s=60, label='Gradient Centroid', edgecolor='k')
    ax[0].scatter(xc, yc, color='red', s=60, label='Refined Center', edgecolor='k')
    ax[0].set_title('ROI with Slopes and Centers')
    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].axis('off')

    # ---------- AX[1]: All lines through midpoints ----------
    ax[1].imshow(roi, cmap='gray', origin='upper')
    L = max(roi.shape)  # line length for plotting
    for i in range(0, mk.shape[0], 2):
        for j in range(0, mk.shape[1], 2):
            m = mk[i, j]
            x0, y0 = xk[i, j], yk[i, j]
            if np.isfinite(m):
                x_line = np.array([x0 - L, x0 + L])
                y_line = m * (x_line - x0) + y0
                ax[1].plot(x_line, y_line, color='orange', alpha=0.8, linewidth=0.5)
    ax[1].scatter(xc, yc, color='red', s=60, edgecolor='k')
    ax[1].set_title('Gradient Lines Focused on Refined Center')
    ax[1].axis('off')
    ax[1].set_xlim(0, L)
    ax[1].set_ylim(0, L)


    # ---------- AX[2]: Roberts gradient magnitude ----------
    Gmag = np.sqrt(Gu**2 + Gv**2)
    im = ax[2].imshow(Gmag, cmap='inferno', origin='upper')
    ax[2].set_title('Roberts Gradient Magnitude Map')
    ax[2].axis('off')
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label='|∇I|')

    plt.show()

def batchRadialSymmetry(video, track_df, length):

    F, H, W = np.shape(video)

    xs = track_df['x'].values
    ys = track_df['y'].values
    frames = track_df['frame'].values

    x_refined = np.zeros_like(xs, dtype=float)
    y_refined = np.zeros_like(ys, dtype=float)

    for i in range(len(xs)):
        frame_index = frames[i]
        xc = int(np.round(xs[i]))
        yc = int(np.round(ys[i]))

        tx_l = max(xc - length // 2, 0)
        tx_r = min(xc + length // 2 + 1, W)
        ty_l = max(yc - length // 2, 0)
        ty_r = min(yc + length // 2 + 1, H)

        roi = video[frame_index, ty_l:ty_r, tx_l:tx_r]

        # Subpixel refinement
        xc_refined, yc_refined = radialSymmetryROI(roi)

        # Convert to global coordinates
        x_refined[i] = tx_l + xc_refined
        y_refined[i] = ty_l + yc_refined

    # update dataframe with refined coordinates ( this might work)
    track_df['x_refined'] = x_refined
    track_df['y_refined'] = y_refined

    # reorder so that refined x and y are at the start of the dataframe
    track_df_cols = track_df.keys()
    org_keys = [track_df_cols[-1], track_df_cols[-2]] + track_df_cols[:-2].tolist()
    ord_track_df = track_df.loc[:, org_keys]

    return ord_track_df
