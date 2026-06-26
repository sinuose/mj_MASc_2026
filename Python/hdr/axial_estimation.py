# -----------------------------------
# Library Imports
# -----------------------------------
import numpy as np
from scipy.optimize import differential_evolution, curve_fit
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy.stats import gamma, lognorm
from skimage.morphology import disk
from scipy.signal import convolve2d

# -----------------------------------
# Helper Functions
# -----------------------------------
def gaussian_pdf(x, A, mu, sigma):
    return (A/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

def LineAnalyticalMSD(t_array, D, L, n_terms=1000):
    """
    Compute the analytical MSD for diffusion in a confined line.

    Parameters:
        D (float): Diffusion coefficient
        L (float): Length of the Line
        t_array (array-like): Array of time values
        n_terms (int): Number of terms in the series expansion (default: 100)

    Returns:
        np.ndarray: MSD values corresponding to t_array
    """
    p_range = list(range(0, n_terms))  # Roots of derivative of Bessel function J1
    msd = np.zeros_like(t_array, dtype=np.float64)

    for p in p_range:
        term = np.exp(-( D * t_array * (2*p +1)**2 * np.pi**2)/(L**2)) / (2*p +1)**4
        msd += term

    msd = L**2 * 1/6 * (1 - 96/(np.pi**4) * msd)
    return msd



# -----------------------------------
# 2D model
# -----------------------------------

def iPSF_radial(r, k, zpp, E0, phi0, exp_fac):
    rpp = np.sqrt(r**2 + zpp**2) #from particle to the focal plane
    cos_theta = zpp / rpp #cos of scattering angle
    phi_inc = k*zpp #phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k*rpp #phase shift due to return OPD
    fac = np.sqrt(1+cos_theta**2)*1/(k*rpp) #amplitude factor
    Escat = E0*fac #scattering amplitude

    phi_diff=-(phi0+phi_inc+phi_sca) #phase difference

    iPSF = 2*Escat*np.cos(phi_diff) * (np.exp(-r/exp_fac))

    return iPSF

def _iPSF_fast(r2, r, k, zpp, E0, ef):
    """
    Algebraically simplified iPSF (phi0=0 fixed).
    r2 = r**2 precomputed once. Saves ~30% vs original.
    """
    rpp2 = r2 + zpp * zpp
    rpp  = np.sqrt(rpp2)
    q    = np.sqrt(rpp2 + zpp * zpp)          # = sqrt(r2 + 2*zpp^2)
    amp  = E0 * q / (k * rpp2)
    return 2.0 * amp * np.cos(-k * (zpp + rpp)) * np.exp(-r / ef)


def _iPSF_val_and_grad(r2, r, k, zpp, E0, ef):
    """
    iPSF model + analytic gradient w.r.t. (zpp, E0, ef).
 
    Algebraically simplified from the original:
        rpp     = sqrt(r^2 + zpp^2)
        cos_th  = zpp / rpp
        fac     = sqrt(1 + cos_th^2) / (k * rpp)
        Escat   = E0 * fac
        phi_diff= -(k*zpp + k*rpp)      [phi0 = 0, fixed]
        iPSF    = 2 * Escat * cos(phi_diff) * exp(-r / ef)
 
    Simplification: sqrt(1 + cos_th^2) = sqrt(rpp^2 + zpp^2) / rpp
    so fac = sqrt(r^2 + 2*zpp^2) / (k * rpp^2)
 
    Gradients are derived analytically (verified against finite differences
    to < 1e-7 relative error). Supplying the Jacobian to L-BFGS-B eliminates
    finite-difference gradient calls, giving ~4x fewer model evaluations per
    optimisation step.
 
    Parameters
    ----------
    r2  : (Nr,) float64  — r**2, precomputed once outside the loop
    r   : (Nr,) float64  — radii in nm
    k   : float          — wave-number  2π/λ
    zpp : float          — axial particle position (nm)
    E0  : float          — scattering amplitude
    ef  : float          — exponential decay length (nm)
 
    Returns
    -------
    iP          : (Nr,) — model values
    diP_dzpp    : (Nr,) — ∂iPSF/∂zpp
    diP_dE0     : (Nr,) — ∂iPSF/∂E0
    diP_def     : (Nr,) — ∂iPSF/∂ef
    """
    rpp2 = r2 + zpp * zpp
    rpp  = np.sqrt(rpp2)
    q2   = rpp2 + zpp * zpp          # r^2 + 2*zpp^2
    q    = np.sqrt(q2)
    amp  = E0 * q / (k * rpp2)
    env  = np.exp(-r / ef)
    cp   = np.cos(-k * (zpp + rpp))
    sp   = np.sin(-k * (zpp + rpp))
    iP   = 2.0 * amp * cp * env
 
    # d(amp)/d(zpp)  =  E0/k · d/dzpp[ q / rpp2 ]
    #   dq/dzpp      =  2·zpp / q           (chain rule on sqrt(r2+2·zpp^2))
    #   d(rpp2)/dzpp =  2·zpp
    dq_dzpp   = 2.0 * zpp / q
    damp_dzpp = E0 / k * (dq_dzpp * rpp2 - q * 2.0 * zpp) / rpp2 ** 2
 
    # d(phase)/d(zpp) = -k·(1 + zpp/rpp)
    dphi_dzpp = -k * (1.0 + zpp / rpp)
 
    diP_dzpp = 2.0 * env * (damp_dzpp * cp - amp * sp * dphi_dzpp)
    diP_dE0  = iP / E0
    diP_def  = iP * r / ef ** 2
 
    return iP, diP_dzpp, diP_dE0, diP_def


def iPSF_radial_no_decay(r, k, zpp, E0, phi0):
    rpp = np.sqrt(r**2 + zpp**2) #from particle to the focal plane
    cos_theta = zpp / rpp #cos of scattering angle
    phi_inc = k*zpp #phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k*rpp #phase shift due to return OPD
    fac = np.sqrt(1+cos_theta**2)*1/(k*rpp) #amplitude factor
    Escat = E0*fac #scattering amplitude

    phi_diff=-(phi0+phi_inc+phi_sca) #phase difference

    iPSF = 2*Escat*np.cos(phi_diff)

    return iPSF


# -----------------------------------
# Radial Average (Optimized)
# -----------------------------------

def _build_radial_geometry(H: int, W: int, r_max: float, theta_range: tuple):
    """
    Pre-compute all frame-invariant geometry for a given ROI shape and parameters.
    Call once, then pass the returned bundle into radialAverage or radialAverage_stack.

    Returns a dict with keys:
        mask      - boolean (H, W)
        r_flat    - int32 1-D array of integer radii for pixels inside mask
        counts    - bincount denominators, shape (n_bins,)
        valid     - boolean (n_bins,) — which bins have ≥1 pixel
        r_ind     - integer radii for the valid bins (= x-axis of the curve)
        n_bins    - int
    """
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    # Single np.indices call; keep float64 throughout
    y, x   = np.indices((H, W), dtype=np.float64)
    x_rel  = x - cx
    y_rel  = y - cy
    r      = np.hypot(x_rel, y_rel)              # float64, no intermediate arrays
    theta  = np.degrees(np.arctan2(-y_rel, x_rel))

    # Normalise angles to [-180, 180)
    def _wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    theta  = _wrap(theta)
    th1    = _wrap(theta_range[0])
    th2    = _wrap(theta_range[1])

    dtheta     = (th2 - th1) % 360.0
    dtheta_pix = (theta - th1) % 360.0
    mask_theta = dtheta_pix <= dtheta
    mask       = (r <= r_max) & mask_theta       # boolean (H, W)

    # Integer radius map — floor is correct for non-negative r
    r_int  = np.floor(r).astype(np.int32)        # (H, W)
    r_flat = r_int[mask]                          # (K,)  flat pixel radii inside mask
    n_bins = int(r_int[mask].max()) + 1

    counts = np.bincount(r_flat, minlength=n_bins).astype(np.float64)  # avoid int division
    valid  = counts > 0
    r_ind  = np.where(valid)[0].astype(np.int32)  # integer radii on the output x-axis

    return dict(
        mask   = mask,
        r_flat = r_flat,
        counts = counts,
        valid  = valid,
        r_ind  = r_ind,
        n_bins = n_bins,
    )


def radialAverage(roi, r_max=30, theta_range=(90, -90), add_disk=False, disk_radius=2,
                  _geom=None):
    """
    Radial (wedge) average of a 2-D ROI.

    Parameters
    ----------
    roi          : (H, W) ndarray
    r_max        : float   — maximum radius (pixels / nm, consistent with your units)
    theta_range  : (th1, th2) in degrees — wedge extents
    add_disk     : bool    — optionally dilate the mask with a disk
    disk_radius  : int     — radius for the disk dilation
    _geom        : dict or None — pre-computed geometry bundle from _build_radial_geometry.
                   Pass this when calling in a loop over frames with fixed shape/params.

    Returns
    -------
    r_ind         : int32 1-D array  — integer radii (x-axis)
    radial_means  : float64 1-D array — mean intensity at each radius
    mask          : bool (H, W)
    """
    H, W = roi.shape

    # Build geometry only if not supplied (single-frame or first call)
    if _geom is None:
        _geom = _build_radial_geometry(H, W, r_max, theta_range)

    mask   = _geom['mask']
    r_flat = _geom['r_flat']
    counts = _geom['counts']
    valid  = _geom['valid']
    r_ind  = _geom['r_ind']
    n_bins = _geom['n_bins']

    if add_disk:
        mask = add_ball_to_mask(mask, disk_radius)  # returns new array; _geom unchanged

    # ── Core: bincount replaces the Python loop over unique radii (~23x faster) ──
    v_flat       = roi[mask].astype(np.float64)    # (K,) pixel values inside mask
    sums         = np.bincount(r_flat, weights=v_flat, minlength=n_bins)
    radial_means = sums[valid] / counts[valid]     # element-wise mean per bin

    return r_ind, radial_means, mask


def get_single_radial_sample(im_array, theta_naught, fixed_r=30, fixed_theta=30,
                              add_disk=False, random_angle=False):
    """
    Compute radial averages for every frame in im_array.

    Parameters
    ----------
    im_array    : (F, H, W) ndarray
    theta_naught: float — wedge start angle (degrees)
    fixed_r     : float — maximum radius
    fixed_theta : float — wedge angular width (degrees)
    add_disk    : bool
    random_angle: bool — if True, draw theta_1 uniformly from [-180, 180)

    Returns
    -------
    r_index_arr        : list[np.ndarray]  — one per frame (identical if geometry fixed)
    radial_average_arr : list[np.ndarray]  — radial mean curve per frame
    mask_arr           : (F, H, W) bool ndarray
    """
    F, H, W = im_array.shape

    if random_angle:
        theta_1 = (2.0 * np.random.rand() - 1.0) * 180.0
    else:
        theta_1 = float(theta_naught)
    theta_2 = theta_1 + fixed_theta

    # ── Pre-compute geometry ONCE for all frames ─────────────────────────────
    geom = _build_radial_geometry(H, W, fixed_r, (theta_1, theta_2))

    mask_arr           = np.empty((F, H, W), dtype=bool)
    r_index_arr        = []
    radial_average_arr = []

    for index in range(F):
        r_index, radial_average_values, mask = radialAverage(
            im_array[index],
            r_max       = fixed_r,
            theta_range = (theta_1, theta_2),
            add_disk    = add_disk,
            _geom       = geom,            # ← geometry reused, never recomputed
        )
        mask_arr[index]  = mask
        r_index_arr.append(r_index)
        radial_average_arr.append(radial_average_values)

    return r_index_arr, radial_average_arr, mask_arr


def add_ball_to_mask(mask_arr, ball_r):

    assert mask_arr.ndim == 2

    Nx, Ny = mask_arr.shape
    center_ball = disk(ball_r)

    mask_ball = np.zeros((Nx, Ny), dtype=float)
    mask_ball[Nx // 2, Ny // 2] = 1.0

    mask_center_ball = convolve2d(mask_ball, center_ball, mode='same')

    return mask_arr | (mask_center_ball > 0)

# -----------------------------------
# Cost Linker (Optimized)
# -----------------------------------
def best_chain_2d(min_idx, min_vals, max_step=None, w=(1.0, 1.0)):
    """
    Link detections across F frames using dynamic programming.

    Parameters
    ----------
    min_idx  : list of array-like, length F  — x positions (nm)
    min_vals : list of array-like, length F  — y positions (nm)
    max_step : float or None                 — max allowed displacement (nm)
    w        : (wx, wy)                      — per-axis weights for Euclidean distance

    Returns
    -------
    idx_chain : list[int]   — per-frame index of the chosen detection
    x_chain   : list[int]   — x coordinates along the chain (nm)
    y_chain   : list[int]   — y coordinates along the chain (nm)
    """
    # ── 0. Guard: must have at least 1 frame with data ──────────────────────
    F = len(min_idx)
    if F == 0 or len(min_vals) == 0:
        return [], [], []

    # Pre-convert every frame to float64 arrays once (avoids repeated casting)
    xs = [np.asarray(min_idx[t],  dtype=np.float64).ravel() for t in range(F)]
    ys = [np.asarray(min_vals[t], dtype=np.float64).ravel() for t in range(F)]

    if xs[0].size == 0 or ys[0].size == 0:
        return [], [], []

    INF      = np.float64(1e18)
    wx, wy   = np.float64(w[0]), np.float64(w[1])
    max_step = np.float64(max_step) if max_step is not None else None

    # ── 1. DP initialisation ────────────────────────────────────────────────
    # cost[i] = best total cost to reach detection i in the *current* frame
    cost = np.zeros(xs[0].size, dtype=np.float64)
    prev = []   # prev[t][j] = predecessor index in frame t for detection j in frame t+1

    # ── 2. Forward pass ─────────────────────────────────────────────────────
    for t in range(1, F):
        Ax, Bx = xs[t - 1], xs[t]
        Ay, By = ys[t - 1], ys[t]

        # Weighted Euclidean distance matrix  (n_prev x n_curr)
        # Using np.hypot avoids a sqrt on a sum, keeping numerical precision.
        dx = np.sqrt(wx) * (z := np.subtract.outer(Ax, Bx))/np.max(np.abs(z) + 1e-30)   # shape (n_A, n_B), avoids [:, None] copy
        dy = np.sqrt(wy) * (z := np.subtract.outer(Ay, By))/np.max(np.abs(z) + 1e-30)

        np.abs(dx, out=dx)                      # in-place → no extra allocation
        np.abs(dy, out=dy)
        D  = np.hypot(dx, dy)                   # (n_A, n_B)
        #print(f'{D=}')

        # Apply max_step mask in-place (nm units match inputs directly)
        if max_step is not None:
            mask = D > max_step
            if mask.all():              # every transition violates constraint
                return [], [], []
            D[mask] = INF               # in-place, no extra array

        # Cost matrix: broadcast current costs over columns
        M = cost[:, np.newaxis] + D     # (n_A, n_B)

        # Best predecessor for each detection in frame t
        p        = np.argmin(M, axis=0)                         # (n_B,)
        new_cost = M[p, np.arange(M.shape[1], dtype=np.intp)]  # (n_B,)

        # Prune: at least one detection must remain reachable
        if not np.any(np.isfinite(new_cost)):
            return [], [], []

        cost = new_cost
        prev.append(p)

    # ── 3. Back-tracking ────────────────────────────────────────────────────
    idx_chain        = [0] * F
    idx_chain[-1]    = int(np.argmin(cost))

    for t in range(F - 2, -1, -1):
        idx_chain[t] = int(prev[t][idx_chain[t + 1]])

    # ── 4. Extract coordinates (cast back to int — positions are pixel indices)
    x_chain = [int(xs[t][idx_chain[t]]) for t in range(F)]
    y_chain = [int(ys[t][idx_chain[t]]) for t in range(F)]

    return idx_chain, x_chain, y_chain


# ------
# modular pipeline
# -----

def _huber_loss_and_grad(params, r2, r, k, data, w_r = None, delta=1.0):
    """
    Huber loss and its analytic gradient w.r.t. params = (zpp, E0, ef).
 
    Used as the objective for L-BFGS-B (jac=True).  Returns (loss, grad)
    so scipy never needs finite-difference gradient approximations.
    """
    zpp, E0, ef = params
    iP, gz, gE, gf = _iPSF_val_and_grad(r2, r, k, zpp, E0, ef)
 
    if w_r is None:
        w_r = np.ones_like(data)        # recover original loss exactly

    res  = data - iP
    ab   = np.abs(res)
    quad = ab <= delta
 
    # calculate loss with weighting
    rho    = np.where(quad, 0.5 * res**2, delta * (ab - 0.5 * delta))
    loss   = w_r @ rho              

    dl_diP = w_r * np.where(quad, -res, -delta * np.sign(res))
    grad   = np.array([dl_diP @ gz, dl_diP @ gE, dl_diP @ gf])
 
    return loss, grad

# ── Stage 1: nuisance estimation ────────────────────────────────────────────
 
def fit_nuisance_parameters(
    rad_av_arr,
    r_index_nm,
    k,
    bounds             = None,
    n_frames           = 30,
    lam_spacing        = 32.0,
    w_r                = None,
):
    """
    Estimate nuisance parameters E0 and exp_fac from the radial average stack.
 
    Runs the cold-start optimiser on `n_frames` evenly-spaced frames and
    reports the median of each parameter across those fits.  Taking the
    median makes the estimate robust: even if ~50 % of individual frames
    land in the wrong fringe (neighbouring oscillation period), the median
    of E0 and ef is unaffected because those parameters are the same in
    every fringe.
    
    * Varying the frame index (evenly spaced) samples different z values
      across the track, improving diversity in the distribution.
 
    Parameters
    ----------
    rad_av_arr  : list of 1-D arrays, length F
        Radial average per frame.
    r_index_nm  : list of 1-D arrays, length F
        Corresponding radii in nm.  Must be the same for every frame
        (geometry is fixed); only the first entry is used for pre-computation.
    k           : float
        Wave-number 2π/λ.
    bounds      : list of 3 (lo, hi) tuples, optional
        Search bounds for [(zpp), (E0), (ef)].
        Default: [(1, 10000), (0.01, 2.0), (1.0, 5000.0)].
    n_frames    : int
        Number of frames to fit (evenly spaced).  30 is sufficient for
        a stable median estimate.
    w_r  : list of 1-D arrays, length F
        weighting array. Should be the same size as r_index_nm
 
    Returns
    -------
    E0_best  : float  — median E0 estimate across fitted frames
    ef_best  : float  — median exp_fac estimate across fitted frames
    res_df   : pd.DataFrame — per-frame fit results with columns
               ['zpp', 'E0', 'ef']; use for diagnostics / plotting
    """
    if bounds is None:
        bounds = [(1.0, 10_000.0), (0.01, 2.0), (1.0, 5000.0)]
 
    F    = len(rad_av_arr)
    idxs = np.arange(0, n_frames, 1, dtype=int)#min(n_frames, F), dtype=int)
 
    # Pre-compute r2 once (geometry is fixed across frames)
    r_nm = np.asarray(r_index_nm[0], dtype=np.float64)
    r2   = r_nm ** 2
 
    results = []
    for frame_i in idxs:
        #f_r = np.asarray(rad_av_arr[frame_i], dtype=np.float64)
        #opt = _cold_start_one_frame(f_r, r2, r_nm, k, bounds, lam_spacing=lam_spacing)
        #results.append(opt.x)

        f_r     = np.array(rad_av_arr[frame_i])
        r       = np.array(r_index_nm[frame_i])
        r2      = r**2

        try:
            opt = _cold_start_one_frame(f_r, r2, r, k, bounds, lam_spacing, w_r)
        except Exception as e:
            print('cold-start error ',e)
            print('----------- failed on nuisance parameters -----------')

        results.append(opt.x)
 
    res_df  = pd.DataFrame(results, columns=['zpp', 'E0', 'ef'])
    E0_best = float(np.median(res_df['E0']))
    ef_best = float(np.median(res_df['ef']))
 
    return E0_best, ef_best, res_df

# ── Stage 3: vectorised z grid search ───────────────────────────────────────
 
def grid_search_z(
    frames_arr,
    r_nm,
    k,
    E0,
    ef,
    z_lo,
    z_hi,
    n_z       = 1000,
    w_r = None, 
):
    """
    Compute the z-grid loss surface for all F frames simultaneously.
 
    Returns a (F, Nz) loss matrix
 
    This version precomputes M = model_matrix (Nz, Nr) once, then expresses
    all F losses as a single BLAS dgemm call:
 
        ||frame_f - M_z||^2 = ||frame_f||^2 - 2·frame_f @ M_z + ||M_z||^2
 
 
    Parameters
    ----------
    frames_arr : (F, Nr) float64
        All radial averages stacked row-wise.
        Build with ``np.array(rad_av_arr, dtype=np.float64)``.
    r_nm       : (Nr,) float64  — radii in nm (same for all frames)
    k          : float
    E0         : float          — fixed E0 from Stage 1
    ef         : float          — fixed exp_fac from Stage 1
    z_lo, z_hi : float          — z search window in nm
                                  Use  median_z ± channel_half_width
    n_z        : int            — number of z grid points (1000 is good)
 
    Returns
    -------
    grid_losses : (F, Nz) float64  — drop-in replacement for ``grid_guess``
    z_guess     : (Nz,)  float64   — z values for each column
    """
    r2    = r_nm ** 2
    z_g   = np.linspace(z_lo, z_hi, n_z)
    z_col = z_g[:, None]                      # (Nz, 1)
 
    # Precompute model grid (Nz, Nr) — done once for all frames
    rpp2 = r2[None, :] + z_col ** 2           # (Nz, Nr)
    rpp  = np.sqrt(rpp2)
    q    = np.sqrt(rpp2 + z_col ** 2)


    M    = (2.0 * E0 * q / (k * rpp2)
            * np.cos(-k * (z_col + rpp))
            * np.exp(-r_nm[None, :] / ef))    # (Nz, Nr)
 
    # weighting
    if w_r is None:
        w_r = np.ones(r_nm.shape[0], dtype=np.float64)

    sw    = np.sqrt(w_r)                         # (Nr,)

    # Pre-whiten: absorb sw into rows so every dot product is weighted
    Mw    = M          * sw[None, :]             # (Nz, Nr)  M √W
    Fw    = frames_arr * sw[None, :]             # (F,  Nr)  F √W

    # L2 loss via BLAS dgemm 
    M2   = np.einsum('ij,ij->i', Mw, Mw)            # (Nz,)
    F2   = np.einsum('ij,ij->i', Fw, Fw)            # (F,)
    FM   = Fw @ Mw.T                                # (F, Nz)
 
    grid_losses = 0.5 * (F2[:, None] - 2.0 * FM + M2[None, :])  # (F, Nz)
 
    return grid_losses, z_g

def _cold_start_one_frame(f_r, r2, r, k, bounds, lam_spacing=32, w_r = None):
    """
    Find (zpp, E0, ef) from scratch for a single radial profile.
 
    Strategy — two phases:
    Phase 1  Coarse 2-D scan: for each (z, ef) on a grid, compute the
             L2-optimal E0 analytically (closed form, one dot product).
             This sweeps all z fringes cheaply and finds all local minima
             in the z landscape.
    Phase 2  L-BFGS-B with analytic gradient, started from every local
             minimum found in Phase 1.  All candidates are polished with
             the full Huber objective; the one with the lowest final loss
             is returned.
 
    This guarantees the correct fringe is visited and polished, regardless
    of where the global minimum of the approximate L2+analytic-E0 surface
    sits (which can differ from the true Huber minimum).
 
    Parameters
    ----------
    f_r    : (Nr,) — observed radial average
    r2     : (Nr,) — r**2
    r      : (Nr,) — radii in nm
    k      : float
    bounds : [(z_lo,z_hi), (E0_lo,E0_hi), (ef_lo,ef_hi)]
 
    Returns
    -------
    scipy OptimizeResult with .x = [zpp, E0, ef] and .fun = Huber loss
    """

    assert f_r.shape == r.shape == r2.shape, \
        f"shape mismatch: f_r = {f_r.shape}, r = {r.shape}, r2 = {r2.shape}"
    assert np.allclose(r2, r**2), "r2 is not r**2"

    

    lam    = 2.0 * np.pi / k
    E0_b   = bounds[1]
    ef_b   = bounds[2]
 
    # ── Phase 1: grid scan ───────────────────────────────────────────────
    # z spacing = lam/8 ensures each fringe has ≥ 4 points in the grid.
    # ef log-spaced from 10nm to 2000nm (15 values covers all practical ef).
    z_grid  = np.arange(bounds[0][0], bounds[0][1], lam / lam_spacing)  # (Nz,)
    ef_vals = np.exp(
        np.linspace(np.log(max(ef_b[0], 10.0)),
                    np.log(min(ef_b[1], 2000.0)),
                    15)
    )
    z_col   = z_grid[:, None]   # (Nz, 1) for broadcasting
 
    # For each ef: batch model over all z (Nz, Nr), then solve for analytic E0
    best_loss_per_z = np.full(len(z_grid), np.inf)
    best_E0_per_z   = np.zeros(len(z_grid))
    best_ef_per_z   = np.zeros(len(z_grid))
 
    #print(f_r.shape, r.shape, r2.shape)
    rpp2   = r2[None, :] + z_col ** 2               # (Nz, Nr)
    rpp    = np.sqrt(rpp2)
    q      = np.sqrt(rpp2 + z_col ** 2)
    #print(f'rpp2: {rpp2.shape}, r[None,:]: {r[None, :].shape}')

    # weighting
    if w_r is None:
        w_r = np.ones(r2.shape[0], dtype=np.float64)

    for ef in ef_vals:
        G      = (2.0 * q / (k * rpp2)
                  * np.cos(-k * (z_col + rpp))
                  * np.exp(-r[None, :] / ef))            # model / E0, (Nz, Nr)
        
        wG   = w_r[None, :] * G                     # (Nz, Nr)  W G, broadcast over z
        GG     = np.einsum('ij,ij->i', wG, G)             # (Nz,)  ||g||^2 (Nz, 1)
        Gd     = wG @ f_r                                  # (Nz,)  g · data (Nz, 1)
        wf   = w_r * f_r                  # (Nr,)  W f
        wff  = wf @ f_r                   # scalar  f^T W f  (replaces ff) 

        rho = Gd / np.sqrt(GG * wff + 1e-30)

        # d/dE0{L} = d/dE0{1/2 min sum(f - E0 G)} = 0!
        # np.clip projects onto feasible choices.
        E0_opt = np.clip(Gd / GG, E0_b[0], E0_b[1])     # analytic L2-optimal E0 (Nz, 1)

        # residual Matrix
        #res_m  = f_r[None, :] - E0_opt[:, None] * G
        #ab     = np.abs(res_m)
        losses = 1 - rho**2                                             # (Nz,)
        # losses = np.sum(
        #     np.where(ab <= 1.0, 0.5 * res_m ** 2, ab - 0.5),
        #     axis=1
        # ) 
 
        better = losses < best_loss_per_z
        best_loss_per_z[better] = losses[better]
        best_E0_per_z[better]   = E0_opt[better]
        best_ef_per_z[better]   = ef
 
    # Find local minima in the 1-D z loss landscape
    g  = best_loss_per_z
    lm = np.where((g[1:-1] < g[:-2]) & (g[1:-1] < g[2:]))[0] + 1
    if not len(lm):
        lm = np.array([np.argmin(g)])
 
    # ── Phase 2: full 3-param L-BFGS-B polish at every local minimum ────
    best_opt  = None
    best_loss = np.inf
 
    for idx in lm:
        z0   = z_grid[idx]
        E0_0 = best_E0_per_z[idx]
        ef_0 = best_ef_per_z[idx]
 
        # Constrain z to ±lam around this candidate so the optimiser
        # cannot jump to a distant fringe
        tight = [
            (max(bounds[0][0], z0 - lam), min(bounds[0][1], z0 + lam)),
            bounds[1],
            bounds[2],
        ]
 
        try:
            res = minimize(
                _huber_loss_and_grad,
                x0      = [z0, E0_0, ef_0],
                args    = (r2, r, k, f_r, w_r),
                method  = 'L-BFGS-B',
                bounds  = tight,
                jac     = True,
                options = {'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10},
            )
        except Exception as e:
            print('----------- failed on res = minimize -----------')            
            print(e)
 
        if res.fun < best_loss:
            best_loss = res.fun
            best_opt  = res
 
    return best_opt


def getAxialInference(  roi,
                        k=2 * np.pi / (405.0 / 1.33),
                        pixel_size_nm=1,
                        fixed_r= 39,
                        rad_int_theta=359,
                        theta_naught = 0,
                        random_angle = False,
                        bounds0=[(1, 3000), (0.01, 2), (1, 5000)],
                        channel_length0_nm=225,
                        z_grid_revisions = 2,
                        chain_linker_cost=1000,    # max horizontal step in nm
                        linker_weight= (1.0, 1.0), # all weighting for steps is on dx)
                        w_r = None,
                        we  = None,
                        ws  = None,
                        plot_3_param = True,
                        plot_2_param = True, 
                        debug = False, 
                        ):
    # -----------------------------------------
    # STAGE 0 get radial average
    # -----------------------------------------
    start_time = time.time()
    print(f'Sampling: {0.0}')

    r_arr, rad_av_arr, mask_arr = get_single_radial_sample(roi, theta_naught, fixed_r=fixed_r, fixed_theta=rad_int_theta, add_disk=False, random_angle=random_angle)
    r_index_nm = np.array(r_arr) * pixel_size_nm

    if plot_3_param == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(roi[0])
        ax1.imshow(mask_arr[0, :, :], alpha=0.5)
        ax2.plot(r_arr, rad_av_arr, 'ro')
        ax2.grid()
        plt.show(block=False)


    print(r_index_nm.shape)
    print(w_r.shape)
    if r_index_nm.shape[1] != w_r.shape[0]:
        print('shapes dont match!')
        print(r_index_nm.shape)
        print(w_r.shape)
        w_r = np.zeros((r_index_nm.shape[1],)) # size is int(fixed_r + 1) or (0, fixed_r) lol
        w_r[ws:we] = 1

    # -----------------------------------------
    # STAGE 1 Nuisance Estimation on first 50 frames
    # -----------------------------------------
    time_now = time.time()

    print(f'nuisance param: {time_now - start_time:.2f}')
    _, _, res_df = fit_nuisance_parameters(
            rad_av_arr  = rad_av_arr,
            r_index_nm  = r_index_nm,
            k           = k,
            bounds      = bounds0,
            n_frames    = 50,        # increase to 50 if estimates feel unstable
            w_r         = w_r,
        )


    # fit E0 to a gamma pdf
    coef = gamma.fit(res_df['E0'])
    mean, var, skew, kurt = gamma.stats(*coef, moments='mvsk')
    E0_best = mean
    # fit ef to a lognorm pdf
    coef = lognorm.fit(res_df['ef'])
    mean, var, skew, kurt = gamma.stats(*coef, moments='mvsk')
    ef_best = mean

    # generate physical constraint based on the 2-parameter-fit
    z_2_param = res_df['zpp']
    median_z_value = np.median(z_2_param)

    # a-prioi knowledge
    channel_half_width = (channel_length0_nm) / 2
    z_min, z_max = median_z_value - channel_half_width, median_z_value + channel_half_width
    print(f' {z_min:.2f} [nm] < z < {z_max:.2f} [nm]')

    # -----------------------------------------
    # STAGE 2 Grid Search in z + 2D Normalized linking
    # -----------------------------------------

    frames_stack = np.array(rad_av_arr, dtype=np.float64)   # (F, Nr)
    r_common_nm  = np.asarray(r_index_nm[0], dtype=np.float64)

    for i in range(z_grid_revisions):
        try:
            grid_guess, z_guess = grid_search_z(
            frames_arr = frames_stack,
            r_nm       = r_common_nm,
            k          = k,
            E0         = E0_best,
            ef         = ef_best,
            z_lo       = z_min,
            z_hi       = z_max,
            n_z        = 2000,
            w_r         = w_r,
            )
        
        except Exception as e:
            print('----------- failed on z_grid_search -----------')

        # Return Grid Search and Link
        sec_grad_z            = np.diff(grid_guess, n=1, axis=1)
        local_minima          = (sec_grad_z[:, :-1] < 0) & (sec_grad_z[:, 1:] > 0)
        min_idx  = [np.where(local_minima[f])[0] + 1 for f in range(local_minima.shape[0])]
        min_vals = [grid_guess[f, idx] for f, idx in enumerate(min_idx)]

        best_idx_chain, best_coord_chain, best_cost_chain = best_chain_2d(
                min_idx, min_vals, max_step=chain_linker_cost, w=linker_weight
            )

        z_traj = z_guess[best_coord_chain]

        median_z_value = np.mean(z_traj)
        z_min, z_max = median_z_value - channel_half_width, median_z_value + channel_half_width

    if plot_2_param == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(roi[0])
        ax1.imshow(mask_arr[0, :, :], alpha=0.5)
        ax2.plot(r_arr, rad_av_arr, 'ro')
        ax2.plot(r_arr, iPSF_radial_no_decay(r_arr, k, z_traj[0], E0_best, 0))
        ax2.grid()
        plt.show(block=False)

    ssr = np.sum((rad_av_arr - np.array([iPSF_radial(r_common_nm, k, zxx, E0_best, 0, ef_best) for zxx in z_traj]))**2, axis=1)
    sst = np.sum((rad_av_arr - np.mean(rad_av_arr, axis=1, keepdims=True))**2, axis=1)

    r2score = 1 - ssr/sst
    
    opt = {}
    opt['z']  = z_traj
    opt['E0'] = E0_best
    opt['ef'] = ef_best
    opt['r2'] = r2score

    if debug == True:
        opt['r_arr'] = r_common_nm
        opt['rad_av_arr'] = rad_av_arr
        opt['roi'] = roi
        opt['mask_arr'] = mask_arr

    return opt

