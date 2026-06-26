import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarianLinker(source, target, max_distance):
    '''
    Hungarian Linking Algorithm,

    need to write description

    Parameters
    :param source: np.array, 1xm [(x0,y0), ... , (xm_ym)]
    :param target: np.array, 1xn [(x0,y0), ... , (xn_yn)]
    :param max_distance: float
    :return:
    target_indicies
    target_distances
    unassigned_targets
    total_cost
    '''
    src_pts = np.asarray(source, dtype=float)
    tgt_pts = np.asarray(target, dtype=float)

    m = 0 if src_pts.size == 0 else src_pts.shape[0]
    n = 0 if tgt_pts.size == 0 else tgt_pts.shape[0]

    if m == 0 and n == 0:
        return [], np.array([], dtype=int), np.array([], dtype=int), np.array([]), 0.0
    if m == 0:
        return [], np.array([], dtype=int), np.arange(n), np.array([]), 0.0
    if n == 0:
        return [], np.arange(m), np.array([], dtype=int), np.array([]), 0.0

    # pairwise squared distances
    diff = src_pts[:, None, :] - tgt_pts[None, :, :]  # (m, n, 2)
    D = np.sum(diff * diff, axis=2)  # (m, n) squared distances

    # cost matrix
    max_cost = max_distance ** 2
    # create copy to remove large costs
    penalty = max_cost + 1e6  # choose large enough penalty
    D_masked = D.copy()
    D_masked[D_masked > max_cost] = penalty  # infeasible links

    # perform linear sum assignment
    row_idx, col_idx = linear_sum_assignment(D_masked)
    # this gives most probable combination of pairs

    # Determine valid assignments: those whose squared dist <= max_cost
    valid_mask = D[row_idx, col_idx] <= max_cost
    assignments = []
    distances = []
    for r, c, ok in zip(row_idx, col_idx, valid_mask):
        if ok:
            assignments.append((int(r), int(c)))
            distances.append(np.sqrt(float(D[r, c])))

    assignments = np.array(assignments, dtype=int) if assignments else np.empty((0, 2), dtype=int)
    distances = np.array(distances, dtype=float)

    assigned_src = assignments[:, 0] if assignments.size else np.array([], dtype=int)
    assigned_tgt = assignments[:, 1] if assignments.size else np.array([], dtype=int)

    unassigned_src = np.setdiff1d(np.arange(m), assigned_src, assume_unique=True)
    unassigned_tgt = np.setdiff1d(np.arange(n), assigned_tgt, assume_unique=True)

    total_cost = float(np.sum(D[assigned_src, assigned_tgt])) if assignments.size else 0.0

    return assignments, unassigned_src, unassigned_tgt, distances, total_cost


def frame_points(df, frame, xy_cols=("x", "y"), frame_col="frame"):
    """
    Extract x,y points for a single frame + bookkeeping info.
    This is mainly ued for creating matrices to pass into HungarianLinker

    Returns
    -------
    points : np.ndarray, shape (m, 2)
        Array of [x, y] for the frame.
    row_idx : np.ndarray, shape (m,)
        Original DataFrame indices corresponding to each point.
    """
    sub = df[df[frame_col] == frame]
    row_idx = sub.index.to_numpy()
    points = sub.loc[row_idx, list(xy_cols)].to_numpy(dtype=float)
    #particle_ids = sub[particle_col].to_numpy() if particle_col in sub.columns else None
    return points, row_idx


def HungarianLinkingWithActiveTracks(track_df,
                                     max_distance = 30,
                                     x_col = 'x_refined',
                                     y_col = 'y_refined',
                                     particle_col = 'particle',
                                     frame_col = 'frame',
                                     memory=3,
                                     debug = False):
    '''

    :param track_df:
    :param max_distance:
    :param x_col:
    :param y_col:
    :param particle_col:
    :param frame_col:
    :param memory:
    :return:
    '''
    # get a list of the frame numbers
    frames = np.sort(track_df['frame'].unique())
    if debug:
        print(frames)

    # initialize particle column
    track_df["particle"] = None
    # set the first frame as all particle iDS
    first_frame = frames.min()
    first_frame_rows = track_df['frame'] == first_frame
    track_df.loc[first_frame_rows, "particle"] = np.arange(first_frame_rows.sum())

    # initialize active tracks dictionary
    active_tracks = {}
    for x, y, pid in track_df.loc[first_frame_rows, [x_col, y_col, particle_col]].values:
        active_tracks[int(pid)] = {
            'last_frame': first_frame,
            'last_position': np.array([x, y]),
            'missed': 0
        }

    # loop over frames and perform linear sum assignment
    # Iterate consecutive frames
    for f_curr in frames[1:]:
        # build current detections
        tgt_pts, tgt_idx = frame_points(track_df, f_curr, xy_cols=(x_col, y_col), frame_col=frame_col)

        # obtain all active track positions
        src_particles = list(active_tracks.keys())
        src_pts = np.array([active_tracks[p]["last_position"] for p in src_particles])

        if debug:
            print('frame: ', f_curr)
            print(active_tracks)

        # perform linear assignment
        target_indices, target_distances, unassigned_targets, cost = hungarianLinker(src_pts, tgt_pts, max_distance)

        matched_targets = set()

        # update linked tracks
        for src_i, tgt_rel_idx in enumerate(target_indices):
            pid = src_particles[src_i]
            if tgt_rel_idx == -1:
                # this is unassigned,  update missed frames list
                active_tracks[pid]['missed'] += 1
                continue

            # Get the absolute index of the target point in ref_track_df
            matched_targets.add(tgt_rel_idx)
            tgt_abs_idx = tgt_idx[tgt_rel_idx]

            # Assign the same particle ID to the matched target
            track_df.loc[tgt_abs_idx, particle_col] = pid

            # update the active tracks
            active_tracks[pid]['last_frame'] = f_curr  # current frame index
            active_tracks[pid]['last_position'] = tgt_pts[tgt_rel_idx]  # update 'src_tgt' consistently
            active_tracks[pid]['missed'] = 0  # missed flag

        # Track the next available particle ID
        next_particle_id = track_df[particle_col].max() + 1

        # Create new tracks for unmatched detections
        unmatched_tgts = [i for i in range(len(tgt_idx)) if i not in matched_targets]  # set union complement 'not in'
        if unmatched_tgts:
            # correct for more than 1 unmatched particle
            new_ids = np.arange(next_particle_id, next_particle_id + len(unmatched_tgts))
            next_particle_id += len(unmatched_tgts)
            # append active tracks with the unassigned tracks
            for new_pid, tgt_rel_idx in zip(new_ids, unmatched_tgts):
                tgt_abs_idx = tgt_idx[tgt_rel_idx]
                track_df.loc[tgt_abs_idx, particle_col] = new_pid
                active_tracks[int(new_pid)] = {
                    "last_frame": f_curr,
                    "last_position": tgt_pts[tgt_rel_idx],
                    "missed": 0
                }

        # Remove tracks that have been missed for too long
        to_remove = [pid for pid, track in active_tracks.items() if track["missed"] > memory]
        for pid in to_remove:
            del active_tracks[pid]

    return track_df


def HungarianLinkingWithActiveTracksOptimized(track_df,
                                             max_distance=30,
                                             x_col='x_refined',
                                             y_col='y_refined',
                                             particle_col='particle',
                                             frame_col='frame',
                                             memory=3,
                                             debug=False):

    # helper to compute next particle id robustly
    def get_next_particle_id(df):
        if df[particle_col].dropna().size == 0:
            return 0
        else:
            return int(df[particle_col].dropna().astype(int).max()) + 1

    # Ensure numeric particle column; start with NaN for clarity
    track_df = track_df.copy()
    track_df[particle_col] = np.nan

    frames = np.sort(track_df[frame_col].unique())
    first_frame = frames.min()
    first_mask = track_df[frame_col] == first_frame
    n0 = first_mask.sum()
    # initialize particle ids for first frame
    track_df.loc[first_mask, particle_col] = np.arange(n0, dtype=int)

    # build active tracks dict (pid -> metadata)
    active_tracks = {}
    for x, y, pid in track_df.loc[first_mask, [x_col, y_col, particle_col]].values:
        pid = int(pid)
        active_tracks[pid] = {
            'last_frame': int(first_frame),
            'last_position': np.array([float(x), float(y)]),
            'missed': 0
        }


    for f_curr in frames[1:]:
        # get target points and their df indices
        tgt_pts, tgt_idx = frame_points(track_df, f_curr, xy_cols=(x_col, y_col), frame_col=frame_col)

        src_particles = sorted(active_tracks.keys())
        if len(src_particles):
            src_pts = np.array([active_tracks[p]['last_position'] for p in src_particles], dtype=float)
        else:
            src_pts = np.empty((0, 2), dtype=float)

        # Use robust hungarian linker
        assignments, unassigned_src, unassigned_tgts, distances, cost = hungarianLinker(src_pts, tgt_pts, max_distance)

        # Assign matches
        matched_tgts = set()
        for (src_rel, tgt_rel), d in zip(assignments, distances):
            pid = src_particles[int(src_rel)]
            tgt_abs_idx = int(tgt_idx[int(tgt_rel)])
            track_df.loc[tgt_abs_idx, particle_col] = int(pid)
            matched_tgts.add(int(tgt_rel))
            # update active track
            active_tracks[pid]['last_frame'] = int(f_curr)
            active_tracks[pid]['last_position'] = tgt_pts[int(tgt_rel)]
            active_tracks[pid]['missed'] = 0

        # increment missed for unassigned src tracks
        for src_rel in unassigned_src:
            pid = src_particles[int(src_rel)]
            active_tracks[pid]['missed'] += 1

        # create new tracks for unmatched detections
        # next_pid = get_next_particle_id(track_df)
        # if len(unassigned_tgts):
        #     for rel_tgt in unassigned_tgts:
        #         pid = next_pid
        #         next_pid += 1
        #         tgt_abs_idx = int(tgt_idx[int(rel_tgt)])
        #         track_df.loc[tgt_abs_idx, particle_col] = int(pid)
        #         active_tracks[int(pid)] = {
        #             'last_frame': int(f_curr),
        #             'last_position': tgt_pts[int(rel_tgt)],
        #             'missed': 0
        #         }

        # prune tracks missed > memory
        to_remove = [pid for pid, tr in active_tracks.items() if tr['missed'] > memory]
        for pid in to_remove:
            del active_tracks[pid]

    # final cast particle column to int where possible
    if track_df[particle_col].notna().any():
        track_df.loc[track_df[particle_col].notna(), particle_col] = track_df.loc[track_df[particle_col].notna(), particle_col].astype(int)
    return track_df




