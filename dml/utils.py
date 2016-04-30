import numpy as np
import os


def filebase(filepath):
    """For a full filepath like '/path/to/some/file.xyz', return 'file'.

    Like `os.path.basename`, except it also strips the file extension.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    fbase : str
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def slice_tile(x_in, idx, length):
    """Extract a padded tile from a matrix, along the first dimension.

    Parameters
    ----------
    x_in : np.ndarray, ndim=2
        2D Matrix to slice.

    idx : int
        Centered index for the resulting tile.

    length : int
        Total length for the output tile.

    Returns
    -------
    z_out : np.ndarray, ndim=2
        The extracted tile.
    """
    start_idx = idx - length / 2
    end_idx = start_idx + length
    tile = np.zeros([length, x_in.shape[1]])
    x_in = np.concatenate([x_in, tile], axis=0)
    if start_idx < 0:
        tile[np.abs(start_idx):, :] = x_in[:end_idx, :]
    elif end_idx > x_in.shape[0]:
        end_idx = x_in.shape[0] - start_idx
        tile[:end_idx, :] = x_in[start_idx:, :]
    else:
        tile[:, :] = x_in[start_idx:end_idx, :]
    return tile


def slice_ndarray(x_in, idx, length, axis=0):
    """Extract a slice from an ndarray, along a given axis.

    Parameters
    ----------
    x_in : np.ndarray
        Array to slice.

    idx : int, 0 < n < x_in.shape[axis] - length
        Index to start the resulting tile.

    length : int
        Total length for the output tile.

    axis : int, default=0
        Axis over which to slice the ndarray.

    Returns
    -------
    z_out : np.ndarray
        The sliced subtensor.
    """
    slice_idxs = [slice(None, ) for n in range(x_in.ndim)]
    slice_idxs[axis] = slice(idx, idx + length)
    return x_in[tuple(slice_idxs)]


def padded_slice_ndarray(x_in, idx, length, axis=0):
    """Extract a padded slice from an ndarray, along a given axis.
    Parameters
    ----------
    x_in : np.ndarray
        Array to slice.

    idx : int, 0 < n < x_in.shape[axis]
        Centered index for the resulting tile.

    length : int
        Total length for the output tile.

    axis : int, default=0
        Axis over which to slice the ndarray.

    Returns
    -------
    z_out : np.ndarray
        The sliced subtensor.
    """
    pad_length = length / 2
    pad_width = [(0, 0) for n in range(x_in.ndim)]
    pad_width[axis] = [pad_length, pad_length + 1]
    x_in = np.pad(x_in, pad_width, mode='constant')
    return slice_ndarray(x_in, idx, length, axis)
