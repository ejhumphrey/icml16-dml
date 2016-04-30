import pytest


import dml.data as D


def test_parse_filename():
    pass


def test_index_directory(data_dir):
    dset = D.index_directory(data_dir)
    assert dset.shape == (50, 5)


@pytest.fixture(scope='module')
def dataset(data_dir):
    return D.index_directory(data_dir)


def test_split_dataset():
    pass


def test_instrument_neighbors(dataset):
    inst = ['akg', 'eg', 'fl1', 'ho', 'kb', 'kfa', 'klb', 'pt']
    nbs = D.instrument_neighbors(dataset)
    assert len(nbs) == len(inst)


def test_pitch_neighbors(dataset):
    nnums = [21, 22, 36, 39, 64, 68, 73, 82, 84]
    nbs = D.pitch_neighbors(dataset)
    assert len(nbs) == len(nnums)


def test_instrument_pitch_neighbors(dataset):
    nbs = D.instrument_pitch_neighbors(dataset)
    assert len(nbs) == 10


def test_slice_cqt(dataset):
    row = dataset.iloc[0]
    n_in = 4
    slicer = D.slice_cqt(row, window_length=n_in)
    x, y = next(slicer)

    # TODO: Return to this (is the shape right?)
    assert x.shape == (1, 1, n_in, 192)
    assert y['instrument'] == row.instrument
    assert y['note_number'] == row.note_number
    assert y['fcode'] == row.fcode


def test_neighbor_stream(dataset):
    n_in = 4
    nbs = D.instrument_neighbors(dataset)
    stream = D.neighbor_stream(
        nbs, dataset, D.slice_cqt, working_size=2, lam=5,
        window_length=n_in)
    data = next(stream)
    assert data['x_in'].shape == (1, 1, n_in, 192)
    assert data['x_same'].shape == (1, 1, n_in, 192)
    assert data['x_diff'].shape == (1, 1, n_in, 192)
