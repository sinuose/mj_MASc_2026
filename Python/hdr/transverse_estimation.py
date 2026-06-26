import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Traj_XY_confinement_to_radius(trajx, trajy, fps = 300, px_size = 55 * 10**(-9), channel_height = 225, visc = 1.006 * 10**(-3),  distance_thresh = 1* 10**(-6)):
    # ------------------------------- lnp pbs -------------------------------#
    # ------------------------------- Get MSD --------------------------------#
    kb = 1.380649 * 10**(-23)

    Dif_lnp_PBS = []
    Sig_lnp_PBS = []
    Rad_lnp_PBS = []

    # good entries
    good_lnp_PBS = np.arange(len(trajx))
    bad_lnp_PBS = []

    hat_r = np.arange(5, channel_height, 0.1) # 0.1 nm steps

    for index, (traj_x_bead_i, traj_y_bead_i) in enumerate(zip(trajx, trajy)):
        # raw trace of particle
        traj_loop = np.array((traj_x_bead_i, traj_y_bead_i)) * 1/(px_size)

        # filtering of particle
        
        # needs to move at least 1 um
        D_ij = np.subtract.outer(traj_x_bead_i, traj_y_bead_i)
        D_ij = np.abs(D_ij)
        thresh = np.max(D_ij)
        
        if thresh <= distance_thresh:
            bad_lnp_PBS.append(index)

        # compute the diffusivity and radius
        (D_msd, alpha_msd) = getDiffusionMSD(traj_loop, 1, 1/fps, max_time=15, plotting=0, xmin=1, xmax=15, dpi_set=100)
        D_msd_m2 = D_msd * (px_size)**2
        
        Dif_lnp_PBS.append(D_msd)
        Sig_lnp_PBS.append(alpha_msd)

        # estimate r1
        r1 = kb * 293 / (6 * np.pi * visc * D_msd_m2) * 10**(9) # nm

        # stuck particles have too large size
        if r1 > channel_height: #nm
            bad_lnp_PBS.append(index)


        # make cost vector and ge tbest solution
        cost_vector = np.array([( r1 / ConfinementDiffusionCorrection(1, r_test, channel_height/2, 'linsup') - r_test ) for r_test in hat_r])
        r_true = hat_r[np.argmin(np.abs(cost_vector))]


        #if r_true < 20:
        #    print(index)

        #print(1/ConfinementDiffusionCorrection(1, r_true, channel_height, 'parr'))


        #r2 = argmin__{hat_r}(hat_r ConfinementDiffusionCorrection(1, r1, channel_height, 'parr') - hat_r)
        #D_par = ConfinementDiffusionCorrection(D_msd, r1, channel_height, 'parr')

        #r2 = kb * 293 / (6 * np.pi * 1.006 * 10**(-3)  * D_par) * 10**(9) # nm

        Rad_lnp_PBS.append(r_true)

    # filter out the bad ones

    filt_lnp_PBS = [x for x in good_lnp_PBS if x not in bad_lnp_PBS]

    #print(f'median lnp pbs radius: {np.median([Rad_lnp_PBS[x] for x in filt_lnp_PBS])}')
    return Rad_lnp_PBS, filt_lnp_PBS

# def getMeanSquaredDisplacement(trajectory):
#         """
#         Computes MSD_i(j) = <|r(t+i) - r(t)|^2>_t
#         using UNWRAPPED coordinates (correct under PBCs).

#         inputs
#             traj (ndim, F)

#         Returns array of shape (F-1)
#         """
#         try:
#             ndim, F = np.shape(trajectory)
#         except Exception as e:
#             print(e)
#             return np.zeros(len(trajectory) - 1)

#         assert ndim <= 3
#         assert 3 <= F

#         msd = np.zeros(F)
#         lag = np.arange(F)

#         traj_vert = np.array(list(zip(*trajectory)))
#         for i in range(F):
#             disp = traj_vert[i:] - traj_vert[:F - i]
#             msd[i] = np.mean(np.sum(disp**2, axis=1))

#         return msd, lag, ndim

def getMeanSquaredDisplacement(trajectory, max_lag=None):
    """
    Time-averaged MSD (unwrapped coords, correct under PBC):

        MSD(i) = < |r(t+i) - r(t)|^2 >_t ,   i = 1, ..., max_lag

    estimated by the overlapping-origins estimator

        MSD(i) = (1/N_i) * sum_{t=0}^{F-1-i} |r(t+i) - r(t)|^2,
        N_i = F - i   pairs available at lag i.

    Parameters
    ----------
    trajectory : (ndim, F) array, ndim in {1,2,3}
    max_lag    : int, 1 <= max_lag <= F-1 (default F-1)

    Returns
    -------
    msd  : (max_lag,) array, msd[k] = MSD(k+1)
    lag  : (max_lag,) int array, lag[k] = k+1
    ndim : int
    """
    r = np.asarray(trajectory, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"expected (ndim, F); got shape {r.shape}")
    ndim, F = r.shape
    if not 1 <= ndim <= 3:
        raise ValueError(f"ndim must be 1..3; got {ndim}")
    if F < 2:
        raise ValueError(f"need F >= 2 frames; got {F}")

    if max_lag is None:
        max_lag = F - 1
    if not 1 <= max_lag <= F - 1:
        raise ValueError(f"max_lag must lie in [1, {F-1}]; got {max_lag}")

    r = r.T                          # (F, ndim): row t is r(t)
    lag = np.arange(1, max_lag + 1)  # i = 1, ..., max_lag   (NO i = 0)
    msd = np.empty(max_lag)

    for k, i in enumerate(lag):
        disp = r[i:] - r[:F - i]     # r(t+i) - r(t); use F-i, NEVER -i
        msd[k] = np.mean(np.sum(disp**2, axis=1))   # mean over N_i = F-i pairs

    return msd, lag, ndim

def getDiffusionMSD(trajectory, pixel_size, exposure_time, max_time=3, plotting=0, xmin = 0, xmax=100, dpi_set=200):
    trajectory_meters = trajectory
    msd, lag, ndim = getMeanSquaredDisplacement(trajectory_meters)
    time_s = exposure_time * lag
    msd_m2 = msd * pixel_size**2

    # MSD diffusion
    from scipy.optimize import curve_fit
    popt, cov = curve_fit(linear, time_s[:max_time], msd_m2[:max_time]) # m^2/s
    D = popt[0]
    alpha = popt[1]

    # optional plotting
    if plotting == 1:
        plt.figure(11, dpi=dpi_set)
        plt.plot(time_s[xmin:xmax], msd_m2[xmin:xmax], label='MSD')
        plt.plot(time_s[xmin:xmax], linear(time_s, *popt)[xmin:xmax], label='Linear Fit')
        plt.legend()
        plt.show(block=False)

    return (D/(2 * ndim), alpha)

def ConfinementDiffusionCorrection(D_conf, a, h, mode='parr'):
    match mode:
        case 'parr':
            term1 = -1.004   * (a/h)
            term2 =  0.418    * (a/h)**3
            term3 = 0.21 * (a/h)**4
            term4 = -0.169   * (a/h)**5
            lambda_inv = 1 + term1 + term2 + term3 + term4
            return D_conf / lambda_inv 
         
        case 'linsup':
            term1 = -9/16   * (a/h)
            term2 =  1/8    * (a/h)**3
            term3 = -45/256 * (a/h)**4
            term4 = -1/16   * (a/h)**5
            lambda_w1 = 1 + term1 + term2 + term3 + term4

            lambda_w2 = 2/lambda_w1-1
            return D_conf * lambda_w2
         
        case 'perp':
            term1 = -9/8    * (a/h)
            term2 =  1/2    * (a/h)**3
            lambda_inv = 1 + term1 + term2
            return D_conf / lambda_inv
        case _:
            return 0

def linear(x, a, b):
    return a * x + b

def compute_covariance_matrix(T):
        """Returns covariance matrix R for Brownian motion (1D, no noise)."""
        R = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1), (T, T), dtype=int)
        return R

def getDiffusionMLE(trajectory, exposure_time):
        """
        Computes the MLE diffusion coefficient for a 3D trajectory array.

        Parameters:
        - traj_array: np.ndarray of shape (ndim, T) T-length
        - exposure_time: float, time step between trajectory points

        Returns:
        - D_mle: np.ndarray of shape (N,), estimated diffusion coefficients
        """
        traj_array = np.array(trajectory)

        ndim, T = traj_array.shape
        # create covariance matrix
        R = compute_covariance_matrix(T)
        R_inv = np.linalg.inv(R)

        sum_proj = 0.0
        for dim in range(ndim):
            x = traj_array[dim, :] # check
            sum_proj += x @ R_inv @ x  # x^T R^{-1} x

        D_mle = sum_proj / (2 * ndim * (T - 1) * exposure_time)

        return D_mle # this is wrong

