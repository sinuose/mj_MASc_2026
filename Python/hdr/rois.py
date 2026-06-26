from scipy.ndimage import shift
import pandas as pd
import numpy as np

def extractRoi(t_df, pid, video, length, use_refined = True):
    '''
    :param t_df: trackpy array of linked trajectories with particle labels ('particle')
    :param pid:  particle index (same as in dataframe)
    :param video: video to be cropped
    :param length: crop length (assumes a square)

    :return: roi_array: dtype np.array()
    '''
    #check that t_df is a dataframe
    if type(t_df) == pd.DataFrame:
        pass
    else:
        return 0

    F, _, _ = np.shape(video)

    group = t_df[t_df['particle'] == pid]
    if group.empty:
        raise ValueError(f"Particle {pid} not found in dataframe.")

    try:
        if use_refined == True:
            xs = group['x_refined'].values
            ys = group['y_refined'].values
        else:
            xs = group['x'].values
            ys = group['y'].values
    except:
        print('No positions in df')
        return 0
    

    frames = group['frame'].astype(int).values

    roi_list = np.zeros((F, length, length))

    for f, x, y in zip(frames, xs, ys):
        xc = int(np.round(x))
        yc = int(np.round(y))

        # fractional offset that moves (x,y) onto integer center (xc,yc)
        dx = xc - x
        dy = yc - y

        # shift expects (shift_y, shift_x)
        frame = video[f].astype(float)
        shifted = shift(frame, shift=(dy, dx), order=3, mode='nearest')

        tx_l = xc - length // 2
        tx_r = tx_l + length
        ty_l = yc - length // 2
        ty_r = ty_l + length

        # compute crop with safe clipping and pad if necessary
        pad_left = max(0, 0 - tx_l)
        pad_top = max(0, 0 - ty_l)
        pad_right = max(0, tx_r - length)
        pad_bottom = max(0, ty_r - length)

        if any((pad_left, pad_top, pad_right, pad_bottom)):
            shifted = np.pad(shifted, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            # adjust crop coords because of padding
            tx_l += pad_left
            tx_r += pad_left
            ty_l += pad_top
            ty_r += pad_top

        roi = shifted[ty_l:ty_r, tx_l:tx_r]

        if roi.shape != (length, length):
            # final safety: resize/pad/truncate to exact shape
            roi = np.pad(roi,
               (((0, max(0, length - roi.shape[0]))), ((0, max(0, length - roi.shape[1])))),
                         mode='edge')[:length, :length]

        roi_list[f] = shifted[ty_l:ty_r, tx_l:tx_r]

    return roi_list
