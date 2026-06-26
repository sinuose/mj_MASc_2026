import cv2
import numpy as np
from hdr.mj_math_utilities import *
from hdr.rvt import rvt

def gaussianBlurBatch(video, ksize, sigmax =1, sigmay = None):
    '''
    Gaussian Blur tool
    :param video: (F, H, W) numpy array
    :param ksize: (X, Y)
    :param sigmaX: float
    :param sigmaY: float
    :return:
        smoothed video
    '''
    if sigmay is None:
        sigmay = sigmax

    output = np.array([cv2.GaussianBlur(x, ksize=ksize, sigmaX=sigmax, sigmaY=sigmay) for x in video])
    return  output

def normalizePerFramePixelValues(video, min =0 , max=255):
    '''

    :param video: video to be normalized
    :param min: lower bound pixel value
    :param max: upper bound pixel value
    :return: normalized video with range in [min, max]
    '''
    normalized = [(x - np.min(x))/(np.max(x) - np.min(x)) * (max - min) + min for x in video]
    return normalized


def fourierBandPassDonut(video, min_f, max_f):
    if np.ndim(video) != 3:
        raise ValueError("Input must be a 3D array (F, H, W)")

    F, H, W = video.shape

    # frequency coordinates
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # radial frequency mask
    r = np.sqrt(uu**2 + vv**2)
    mask = (r >= min_f) & (r <= max_f)
    mask3d = np.repeat(mask[None, :, :], F, axis=0)

    # FFT band-pass
    video_fft = np.fft.fft2(video, axes=(1, 2))
    video_fft_shift = np.fft.fftshift(video_fft, axes=(1, 2))
    video_fft_shift *= mask3d
    video_bp = np.fft.ifft2(np.fft.ifftshift(video_fft_shift, axes=(1, 2)), axes=(1, 2)).real

    return video_bp, mask, video_fft_shift



def Nov2RVTpreprocess(video, rmin, rmax_arr, fmin=0.65, fmax=1, kind='basic'):
    if np.ndim(video) != 3:
        print("arr must be 3D")
        return 0

    rvt_scan_arr = []
    for i in range(len(rmax_arr)):
        rvt_scan_arr.append(rvt(video, rmin, rmax_arr[i], kind=kind)) # (L, F, X, Y)

    rvt_scan_sum = np.sum(rvt_scan_arr, axis=0) # ( F, X, Y)
    rvt_scan_sum_normal = np.abs(rvt_scan_sum)

    # normalize to [0,1]
    rvt_scan_sum_normal_unity = (rvt_scan_sum_normal - np.min(rvt_scan_sum_normal, axis=0)) / (np.max(rvt_scan_sum_normal, axis = 0) - np.min(rvt_scan_sum_normal, axis = 0 ))

    # square root twice
    rvt_scan_sum_root = np.sqrt(np.sqrt(rvt_scan_sum_normal_unity))

    #
    ft_rvt_norm, _,_ = fourierBandPassDonut(rvt_scan_sum_root, fmin, fmax)

    return ft_rvt_norm # (F, X, Y)


def fourierBandPassDonut_gpu(video, min_f, max_f, device='cuda'):
    """
    GPU-accelerated band-pass using CuPy if available; falls back to NumPy on CPU.
    Input: video (F, H, W) numpy float32/float64
    Returns: (video_bp, mask, video_fft_shift) as numpy arrays
    """
    import numpy as np

    # Try CuPy
    xp = np
    cupy_ok = False
    if device == 'cuda':
        try:
            import cupy as cp
            _ = cp.cuda.runtime.getDeviceCount()  # raises if no GPU
            xp = cp
            cupy_ok = True
        except Exception:
            print('error')
            return 0

    if np.ndim(video) != 3:
        raise ValueError("Input must be a 3D array (F, H, W)")

    # Convert to appropriate array module
    # Keep precision (float32 is often faster; cast if your pipeline allows)
    video_x = xp.asarray(video, dtype=video.dtype, order='C')

    F, H, W = video_x.shape

    # Frequency coordinates on (H, W); these are small arrays so building on GPU is fine
    u = xp.fft.fftfreq(W)
    v = xp.fft.fftfreq(H)
    uu, vv = xp.meshgrid(u, v, indexing="xy")

    # Radial mask (avoid repeat by broadcasting later)
    r = xp.sqrt(uu**2 + vv**2)
    mask = (r >= min_f) & (r <= max_f)
    video_fft = xp.fft.fft2(video_x, axes=(1, 2))
    video_fft_shift = xp.fft.fftshift(video_fft, axes=(1, 2))

    # Apply mask with broadcasting (F,1,1) * (H,W) -> (F,H,W)
    video_fft_shift *= mask[None, :, :]
    video_bp = xp.fft.ifft2(xp.fft.ifftshift(video_fft_shift, axes=(1, 2)), axes=(1, 2)).real

    # Return to CPU as numpy arrays so the rest of your code keeps working
    if cupy_ok:
        video_bp_np = xp.asnumpy(video_bp)
        mask_np = xp.asnumpy(mask)
        video_fft_shift_np = xp.asnumpy(video_fft_shift)
        # Optional: free GPU memory
        del video_x, video_fft, video_fft_shift, video_bp, uu, vv, r, mask
        xp.get_default_memory_pool().free_all_blocks()
        return video_bp_np, mask_np, video_fft_shift_np
    else:
        # CPU path with NumPy: identical to your original but avoids repeat
        return video_bp, mask, video_fft_shift
