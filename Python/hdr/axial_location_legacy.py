import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------------------------------
# DO NOT USE
# ------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------
# iPSF models
# ------------------------------------------------------------------------------------------------------------------------

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

def iPSF_radial_with_decay(r, k, zpp, E0, phi0, exp_fac):
    rpp = np.sqrt(r**2 + zpp**2) #from particle to the focal plane
    cos_theta = zpp / rpp #cos of scattering angle
    phi_inc = k*zpp #phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k*rpp #phase shift due to return OPD
    fac = np.sqrt(1+cos_theta**2)*1/(k*rpp) #amplitude factor
    Escat = E0*fac #scattering amplitude

    phi_diff=-(phi0+phi_inc+phi_sca) #phase difference

    iPSF = 2*Escat*np.cos(phi_diff) * (np.exp(-r/exp_fac))

    return iPSF

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

# ------------------------------------------------------------------------------------------------------------------------
# helper to add ball
# ------------------------------------------------------------------------------------------------------------------------

def add_ball_to_mask(mask_arr, ball_r):
    from skimage.morphology import disk
    from scipy.signal import convolve2d

    assert mask_arr.ndim == 2

    Nx, Ny = mask_arr.shape
    center_ball = disk(ball_r)

    mask_ball = np.zeros((Nx, Ny), dtype=float)
    mask_ball[Nx // 2, Ny // 2] = 1.0

    mask_center_ball = convolve2d(mask_ball, center_ball, mode='same')

    return mask_arr | (mask_center_ball > 0)

# ------------------------------------------------------------------------------------------------------------------------
# Radial Average function
# ------------------------------------------------------------------------------------------------------------------------

def radialAverage(roi, r_max=30, theta_range = (90,-90), add_disk=False, disk_radius=2):
    '''
    :param roi:
    :param r_max:
    :param theta_range:
    :return r_ind: the emperical x values of the radial average
    :return radial_means: Radial averaging function f(r)
    :return mask: mask of where the radial average is applied
    '''
    # these are easily callable
    H, W = np.shape(roi)

    # pre-compute the constants for a given frame;

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    # Coordinate grid
    y, x = np.indices((H,W))
    x_rel, y_rel = x - cx, y - cy
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.degrees(np.arctan2(-y_rel, x_rel))

    # Restrict to wedge + radius
    # handle wrapping
    th1, th2 = theta_range

    # normalize to [-180, 180)
    def wrap(a):
        return (a + 180) % 360 - 180

    theta = wrap(theta)
    th1 = wrap(th1)
    th2 = wrap(th2)

    # angular distance from th1 → th2 (wrapped, positive)
    dtheta = (th2 - th1) % 360
    dtheta_pix = (theta - th1) % 360

    mask_theta = dtheta_pix <= dtheta
    mask = (r <= r_max) & mask_theta

    if add_disk == True:
        mask = add_ball_to_mask(mask, disk_radius)

    # compute integer radius map for entire image (same shape as `r`)
    r_int = np.floor(r).astype(int)   # or r.astype(int) — same for non-negative r

    # unique integer radii that actually appear inside the wedge
    unique_r = np.unique(r_int[mask])
    unique_r = np.sort(unique_r)

    r_ind = unique_r  # these are integer radii

    # compute mean for each integer radius
    radial_means = np.empty(unique_r.size, dtype=float)
    for i, rr in enumerate(unique_r):
        sel = (mask) & (r_int == rr)
        radial_means[i] = roi[sel].mean()


    return r_ind, radial_means, mask

# ------------------------------------------------------------------------------------------------------------------------
# linear average for testing
# ------------------------------------------------------------------------------------------------------------------------

def linearAverage(roi, r_max=30):
    '''
    :param roi:
    :param r_max:
    :param theta_range:
    :return r_ind: the emperical x values of the radial average
    :return radial_means: Radial averaging function f(r)
    :return mask: mask of where the radial average is applied
    '''
    # these are easily callable
    H, W = np.shape(roi)

    # pre-compute the constants for a given frame;

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    # Coordinate grid
    y, x = np.indices((H,W))
    x_rel, y_rel = x - cx, y - cy
    r = np.sqrt(x_rel**2 + y_rel**2)

    linear_x_mask = (x == 60)
    linear_y_mask = 1
    linear_mask = linear_x_mask# * linear_y_mask
    mask = linear_mask#(r <= r_max)

    # compute integer radius map for entire image (same shape as `r`)
    r_int = np.floor(r).astype(int)   # or r.astype(int) — same for non-negative r

    # unique integer radii that actually appear inside the wedge
    unique_r = np.unique(r_int[mask])
    unique_r = np.sort(unique_r)

    r_ind = unique_r  # these are integer radii

    # compute mean for each integer radius
    radial_means = np.empty(unique_r.size, dtype=float)
    for i, rr in enumerate(unique_r):
        sel = (mask) & (r_int == rr)
        radial_means[i] = roi[sel].mean()


    return r_ind, radial_means, mask

# ------------------------------------------------------------------------------------------------------------------------
# radial average sampler
# ------------------------------------------------------------------------------------------------------------------------

def get_single_radial_sample(im_array, fixed_r= 30, fixed_theta = 30, add_disk=True):
    mask_arr = np.zeros_like(im_array)
    r_index_arr = []
    radial_average_arr = []
    theta_1 = (2 * np.random.rand() - 1 ) * 180
    theta_2 = theta_1 + fixed_theta
    for index in range(im_array.shape[0]):
        r_index, radial_average_values, mask = radialAverage(im_array[index], r_max=fixed_r, theta_range=[theta_1, theta_2], add_disk=add_disk)
        #r_index, radial_average_values, mask = linearAverage(im_array[index], r_max=fixed_r)

        # append information
        mask_arr[index] = mask
        r_index_arr.append(r_index)
        radial_average_arr.append(radial_average_values)

    return r_index_arr, radial_average_arr, mask_arr


def twostep_axial_estimation_wrapper(
    roi,
    k,
    PIXEL_SIZE_NM,
    bounds,
    fixed_r=55,
    fixed_theta=35,
    add_disk=False,
    dz_refine=0.1,       # nm resolution
    z_window=5.0,        # nm half-window around coarse minimum
):
    """
    Returns:
      trace_df: one row per frame (tidy data)
      summary_df: one row per ROI (aggregate metrics)
    Optionally appends/writes to:
      {out_dir}/{base_name}_trace.csv
      {out_dir}/{base_name}_summary.csv
    """

    # ----------------------------
    # setup run id + output paths
    # ----------------------------
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    trace_path = summary_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / f"{base_name}_trace.csv"
        summary_path = out_dir / f"{base_name}_summary.csv"

    # ----------------------------
    # helper: safe json encoding
    # ----------------------------
    def _to_jsonable(x):
        try:
            return json.dumps(x)
        except TypeError:
            return json.dumps(str(x))

    bounds_json = _to_jsonable(bounds)

    # ----------------------------
    # loss functions (as in your code)
    # ----------------------------
    try:
        def f_obj_huber_fixed_grid(zpp, rv, data, delta=1.0):
            model = iPSF_radial_no_decay(
                rv[None, :],
                k,
                zpp[:, None],
                E0_best,
                phi0_best
            )
            r = data[None, :] - model
            abs_r = np.abs(r)
            loss = np.where(
                abs_r <= delta,
                0.5 * r ** 2,
                delta * (abs_r - 0.5 * delta)
            )
            return np.sum(loss, axis=1)

        def f_obj_huber(params, rv, data, delta=1.0):
            zpp, E0, phi0 = params
            r = data - iPSF_radial_no_decay(rv, k, zpp, E0, phi0)
            abs_r = np.abs(r)
            quad = 0.5 * r ** 2
            lin = delta * (abs_r - 0.5 * delta)
            return np.sum(np.where(abs_r <= delta, quad, lin))

    except Exception as e:
        # If something goes wrong building the functions, fail gracefully
        trace_df = pd.DataFrame([{
            "run_id": run_id, "pid": pid, "roi_idx": roi_idx,
            "status": "error", "error": f"loss_function_build_failed: {repr(e)}"
        }])
        summary_df = trace_df.copy()
        if trace_path is not None and summary_path is not None:
            _append_csv(trace_df, trace_path, append=append)
            _append_csv(summary_df, summary_path, append=append)
        return trace_df, summary_df

    # ----------------------------
    # analysis start (wrapped to be failure-tolerant)
    # ----------------------------
    status = "ok"
    error_msg = ""

    try:
        r_arr, rad_av_arr, mask_arr = get_single_radial_sample(
            roi, fixed_r=fixed_r, fixed_theta=fixed_theta, add_disk=add_disk
        )
        r_index_nm = np.array(r_arr) * PIXEL_SIZE_NM

        # ----------------------------
        # 3-param fit per frame
        # ----------------------------
        best_param = []
        loss_3param = []

        for r_id, f_r in zip(r_index_nm, rad_av_arr):
            res = differential_evolution(
                func=f_obj_huber,
                args=(r_id, f_r),
                bounds=bounds,
                disp=False,
                maxiter=50000,
                tol=1e-12,
                polish=True,
                seed=seed
            )
            best_param.append(res.x)
            loss_3param.append(res.fun)

        res_df = pd.DataFrame(best_param, columns=["zpp", "E0", "phi0"])

        # estimate E0, phi0 (global)
        E0_dist = res_df["E0"].to_numpy()
        phi0_dist = res_df["phi0"].to_numpy()

        E0_best = float(np.mean(E0_dist))
        phi0_best = float(np.mean(phi0_dist))

        E0_mean, E0_std = float(np.mean(E0_dist)), float(np.std(E0_dist))
        phi0_mean, phi0_std = float(np.mean(phi0_dist)), float(np.std(phi0_dist))

        # ----------------------------
        # 1-param coarse search
        # ----------------------------
        z_guess = np.linspace(100, 5000, num=(5000 - 100))
        grid_guess = np.empty((len(r_index_nm), len(z_guess)), dtype=float)

        for i, (r_id, f_r) in enumerate(
            tqdm(zip(r_index_nm, rad_av_arr),
                 total=len(r_index_nm),
                 desc=f"Computing loss landscape (pid={pid})")
        ):
            grid_guess[i] = f_obj_huber_fixed_grid(z_guess, r_id, f_r, delta=1.0)

        loss_coarse = np.min(grid_guess, axis=1)
        idx_coarse = np.argmin(grid_guess, axis=1)
        z_coarse = z_guess[idx_coarse]

        # ----------------------------
        # refinement around each coarse minimum
        # ----------------------------
        z_refined = np.empty(len(z_coarse), dtype=float)
        loss_refined = np.empty(len(z_coarse), dtype=float)

        for i, (r_id, f_r, z0) in enumerate(
            tqdm(zip(r_index_nm, rad_av_arr, z_coarse),
                 total=len(z_coarse),
                 desc=f"Refining minima (pid={pid})")
        ):
            z_local = np.arange(z0 - z_window, z0 + z_window + dz_refine, dz_refine)
            loss_local = f_obj_huber_fixed_grid(z_local, r_id, f_r, delta=1.0)
            j = int(np.argmin(loss_local))
            z_refined[i] = float(z_local[j])
            loss_refined[i] = float(loss_local[j])

        # final z-trace
        z_trace = z_refined
        dz_step = np.r_[np.nan, np.diff(z_trace)]

        # ----------------------------
        # Build trace dataframe (tidy)
        # ----------------------------
        n_frames = len(z_trace)
        frame_idx = np.arange(n_frames)

        trace_df = pd.DataFrame({
            "run_id": run_id,
            "pid": pid,
            "roi_idx": roi_idx,
            "frame_idx": frame_idx,

            # ROI settings
            "fixed_r_px": fixed_r,
            "fixed_theta_deg": fixed_theta,
            "add_disk": add_disk,

            # fit settings
            "k": k,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "bounds_json": bounds_json,
            "dz_refine_nm": dz_refine,
            "z_window_nm": z_window,
            "seed": seed,

            # outputs per frame
            "z_coarse_nm": z_coarse.astype(float),
            "loss_coarse": loss_coarse.astype(float),
            "z_refined_nm": z_refined.astype(float),
            "loss_refined": loss_refined.astype(float),
            "dz_step_nm": dz_step.astype(float),

            # per-frame score from 3-param stage
            "loss_3param": np.array(loss_3param, dtype=float),

            # global fitted values (repeated for convenience)
            "E0_best": E0_best,
            "phi0_best": phi0_best,
            "E0_mean": E0_mean,
            "E0_std": E0_std,
            "phi0_mean": phi0_mean,
            "phi0_std": phi0_std,

            # status bookkeeping
            "status": "ok",
            "error": ""
        })

        # ----------------------------
        # Summary dataframe (one row per ROI)
        # ----------------------------
        summary_df = pd.DataFrame([{
            "run_id": run_id,
            "pid": pid,
            "roi_idx": roi_idx,
            "fixed_r_px": fixed_r,
            "fixed_theta_deg": fixed_theta,
            "add_disk": add_disk,
            "k": k,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "bounds_json": bounds_json,
            "dz_refine_nm": dz_refine,
            "z_window_nm": z_window,
            "seed": seed,

            "n_frames": int(n_frames),

            "E0_best": E0_best,
            "phi0_best": phi0_best,
            "E0_mean": E0_mean,
            "E0_std": E0_std,
            "phi0_mean": phi0_mean,
            "phi0_std": phi0_std,

            # z-trace aggregate stats
            "z_mean": float(np.mean(z_trace)),
            "z_std": float(np.std(z_trace)),
            "z_min": float(np.min(z_trace)),
            "z_max": float(np.max(z_trace)),
            "z_range": float(np.max(z_trace) - np.min(z_trace)),

            # quality scores
            "loss_refined_mean": float(np.mean(loss_refined)),
            "loss_refined_median": float(np.median(loss_refined)),
            "loss_coarse_mean": float(np.mean(loss_coarse)),
            "loss_3param_mean": float(np.mean(loss_3param)),

            "status": "ok",
            "error": ""
        }])

    except Exception as e:
        status = "error"
        error_msg = repr(e)

        trace_df = pd.DataFrame([{
            "run_id": run_id, "pid": pid, "roi_idx": roi_idx,
            "fixed_r_px": fixed_r, "fixed_theta_deg": fixed_theta, "add_disk": add_disk,
            "k": k, "pixel_size_nm": PIXEL_SIZE_NM, "bounds_json": bounds_json,
            "dz_refine_nm": dz_refine, "z_window_nm": z_window, "seed": seed,
            "status": status, "error": error_msg
        }])

        summary_df = trace_df.copy()

    # ----------------------------
    # optional: write/append to csv
    # ----------------------------
    if trace_path is not None and summary_path is not None:
        _append_csv(trace_df, trace_path, append=append)
        _append_csv(summary_df, summary_path, append=append)

    return trace_df, summary_df


def _append_csv(df: pd.DataFrame, path: Path, append: bool = True):
    """
    Append to CSV safely:
    - if append and file exists -> append without header
    - else -> write with header
    """
    path = Path(path)
    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)
