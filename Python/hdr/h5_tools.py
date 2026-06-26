import h5py



def append_entry(h5_path, file_id, entry_id, x, y, z, E0, ef, r2score):
    with h5py.File(h5_path, "a") as f:
        file_group = f.require_group(f"file_{file_id}")
        group_name = f"PID_{entry_id}"
        # overwrite protection
        if group_name in file_group:
            del file_group[group_name]
            # make the row
        entry_group = file_group.create_group(group_name)

        # Scalars
        entry_group.create_dataset("E0", data=E0)
        entry_group.create_dataset("ef", data=ef)

        # Time series (chunked + compressed)
        entry_group.create_dataset("x", data=x, compression="gzip", chunks=True)
        entry_group.create_dataset("y", data=y, compression="gzip", chunks=True)
        entry_group.create_dataset("z", data=z, compression="gzip", chunks=True)
        entry_group.create_dataset("r2score", data=r2score, compression="gzip", chunks=True)

        try:
            assert len(x) == len(y) == len(z)
            entry_group.attrs["frames"] = len(x)
        except Exception as e:
            print('Length of frames is not equal, setting to transvere length')
            entry_group.attrs["frames"] = len(x)
        entry_group.attrs["units"] = "nm"
        entry_group.attrs["version"] = "v1.0"
