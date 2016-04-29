"""Methods and routines to manipulate timbre data."""

import biggie
import itertools
import glob
import numpy as np
import os
import pandas as pd
import pescador
import time

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


def create_entity(npz_file, dtype=np.float32):
    """Create an entity from the given file.

    Parameters
    ----------
    npz_file: str
        Path to a 'npz' archive, containing at least a value for 'cqt'.

    dtype: type
        Data type for the cqt array.

    Returns
    -------
    entity: biggie.Entity
        Populated entity, with the following fields:
            {cqt, time_points, icode, note_number, fcode}.
    """
    (icode, note_number,
        fcode) = [np.array(_) for _ in parse_filename(npz_file)]
    entity = biggie.Entity(icode=icode, note_number=note_number,
                           fcode=fcode, **np.load(npz_file))
    entity.cqt = entity.cqt.astype(dtype)
    return entity


def slice_cqt_entity(entity, length, idx=None):
    """Return a windowed slice of a cqt Entity.

    Parameters
    ----------
    entity : Entity, with at least {cqt, icode} fields
        Observation to window.
        Note that entity.cqt is shaped (num_channels, num_frames, num_bins).

    length : int
        Length of the sliced array.

    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Returns
    -------
    sample: biggie.Entity with fields {cqt, label}
        The windowed observation.
    """
    idx = np.random.randint(entity.cqt.shape[1]) if idx is None else idx
    cqt = np.array([utils.slice_tile(x, idx, length) for x in entity.cqt])
    return biggie.Entity(cqt=cqt, icode=entity.icode,
                         note_number=entity.note_number, fcode=entity.fcode)


def slice_embedding_entity(entity, length, idx=None):
    """Return a windowed slice of a cqt Entity.

    Parameters
    ----------
    entity : Entity, with at least {cqt, icode} fields
        Observation to window.
        Note that entity.cqt is shaped (num_channels, num_frames, num_bins).
    length : int
        Length of the sliced array.
    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Returns
    -------
    sample: biggie.Entity with fields {cqt, label}
        The windowed observation.
    """
    idx = np.random.randint(entity.embedding.shape[0]) if idx is None else idx
    return biggie.Entity(
        embedding=entity.embedding[idx],
        time=entity.time_points[idx],
        fcode=entity.fcode,
        note_number=entity.note_number,
        icode=entity.icode)


def neighbor_stream(dframe, neighbors, slice_func, **kwargs):
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
    pass
