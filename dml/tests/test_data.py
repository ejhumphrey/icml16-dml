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
