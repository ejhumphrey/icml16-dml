"""Methods and routines to manipulate timbre data."""

import biggie
import glob
import numpy as np
import os
import pandas as pd
import pescador
import random

import dml.utils as utils


def parse_filename(fname, sep='_'):
    """Parse a VSL filename into its constituent components.

    Parameters
    ----------
    fname : str
        Filename to parse

    Returns
    -------
    icode, note_number, fcode : str
        Instrument code, note number, and file-ID, respectively.
    """
    i, n, f = utils.filebase(fname).split(sep)[:3]
    return i, int(n), f


def index_directory(base_dir, sep='_'):
    """Index a directory of VSL feature files as a pandas DataFrame.

    Parameters
    ----------
    base_dir : str
        Directory to index.

    Returns
    -------
    dset : pd.DataFrame
        Data frame of features.
    """
    records = []
    index = []
    for f in glob.glob(os.path.join(base_dir, "*.npz")):
        icode, nnum, fcode = parse_filename(f, sep)
        records += [dict(features=f, instrument=icode,
                         note_number=nnum, fcode=fcode,
                         inst_note="{}_{}".format(icode, nnum))]
        index += [sep.join([str(x) for x in parse_filename(f, sep)])]

    return pd.DataFrame.from_records(records, index=index)


def split_dataset(dframe, ratio=0.5):
    """Split dataset into two disjoint sets.

    Parameters
    ----------
    dframe : pd.DataFrame
        Dataset to split.

    Returns
    -------
    set1, set2 : pd.DataFrames
        Partitioned sets.
    """
    index = np.array(dframe.index.tolist())
    np.random.shuffle(index)

    tr_size = int(ratio * len(index))
    tr_index = index[:tr_size]
    te_index = index[tr_size:]

    return dframe.loc[tr_index], dframe.loc[te_index]


def instrument_neighbors(dframe):
    """Create lists of instrument neighbors.

    Parameters
    ----------
    dframe : pd.DataFrame
        Dataframe of the sample index.

    Returns
    -------
    neighbors : dict of lists
        Map of instrument keys to lists of related indexes.
    """
    return {i: dframe[dframe.instrument == i].index.tolist()
            for i in dframe.instrument.unique()}


def pitch_neighbors(dframe):
    """Create lists of instrument/pitch neighbors.

    Parameters
    ----------
    dframe : pd.DataFrame
        Dataframe of the sample index.

    Returns
    -------
    neighbors : dict of lists
        Map of note numbers to lists of related indexes.
    """
    return {nn: dframe[dframe.note_number == nn].index.tolist()
            for nn in dframe.note_number.unique()}


def instrument_pitch_neighbors(dframe, pitch_delta=0):
    """Create lists of instrument/pitch neighbors.

    Parameters
    ----------
    dframe : pd.DataFrame
        Dataframe of the sample index.

    pitch_delta : int, default=0
        Number of pitch values (plus/minus) for forming neighborhoods.

    Returns
    -------
    neighbors : dict of lists
        Map of instrument/pitch keys to lists of related indexes.
    """
    neighbors = dict()
    for i in dframe.instrument.unique():
        for nn in dframe.note_number.unique():
            key = "{}_{}".format(i, nn)
            nidx = np.abs(dframe.note_number - nn) <= 2
            iidx = (dframe.instrument == i)
            c = (nidx & iidx).sum()
            if c > 0:
                neighbors[key] = dframe[nidx & iidx].index.tolist()
    return neighbors


def slice_features(row, field, window_length):
    """Generate slices of CQT observations.

    Parameters
    ----------
    row : pd.Series
        Row from a dataframe.

    window_length : int
        Length of the CQT slice in time.

    Yields
    ------
    x_obs : np.ndarray
        Slice of cqt data.

    meta : dict
        Metadata corresponding to the observation.
    """

    data = np.load(row.features)[field]
    num_obs = data.shape[1] - window_length
    idx = np.random.permutation(num_obs) if num_obs > 0 else None
    np.random.shuffle(idx)
    counter = 0
    meta = dict(instrument=row.instrument, note_number=row.note_number,
                fcode=row.fcode)

    while num_obs > 0:
        n = idx[counter]
        obs = utils.padded_slice_ndarray(data, n, length=window_length, axis=1)
        meta['idx'] = n
        yield obs, meta
        counter += 1
        if counter >= len(idx):
            np.random.shuffle(idx)
            counter = 0


def neighbor_stream(neighbors, dataset, slice_func,
                    working_size=10, lam=25,
                    **kwargs):
    """Produce a sample stream of positive and negative examples.

    Parameters
    ----------
    dframe : pd.DataFrame
        DataFrame to sample from.

    index : str
        Index into the dataframe from which to draw samples.

    neighbors : dict of lists
        Map of neighborhood keys (names) to lists of related indexes.

    slice_func : callable
        Method for slicing observations from a npz archive.

    kwargs : dict
        Keyword arguments to pass through to the slicing function.

    Yields
    ------
    x_obs, x_same, x_diff : np.ndarrays
        Tensors corresponding to the base observation, a similar datapoint,
        and a different one.
    """
    streams = dict()
    for key, indexes in neighbors.items():
        seed_pool = [pescador.Streamer(slice_func, dataset.loc[idx], **kwargs)
                     for idx in indexes]
        streams[key] = pescador.mux(seed_pool, n_samples=None,
                                    k=working_size, lam=lam)
    while True:
        keys = list(streams.keys())
        idx = random.choice(keys)
        x_obs, y_obs = next(streams[idx])
        x_same, y_same = next(streams[idx])
        keys.remove(idx)
        idx = random.choice(keys)
        x_diff, y_diff = next(streams[idx])
        yield dict(x_obs=x_obs, x_same=x_same, x_diff=x_diff)
        # , (y_obs, y_same, y_diff)

# def class_stream2(dframe, n_in, batch_size, working_size=100, lam=20):

#     return pescador.buffer_batch(stream, buffer_size=batch_size)
