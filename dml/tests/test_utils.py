import pytest

import numpy as np

import dml.utils as utils


def __eq(a, b):
    assert a == b


def test_filebase():
    fnames = ['y', 'y.z', 'x/y.z', 'x.y.z']
    results = ['y', 'y', 'y', 'x.y']
    for fn, res in zip(fnames, results):
        yield __eq, utils.filebase(fn), res


def test_slice_ndarray():
    x_in = np.arange(64).reshape(2, 4, 8)
    np.testing.assert_equal(
        utils.slice_ndarray(x_in, idx=0, length=1, axis=0),
        x_in[:1, ...])

    np.testing.assert_equal(
        utils.slice_ndarray(x_in, idx=0, length=1, axis=1),
        x_in[:, :1, :])

    np.testing.assert_equal(
        utils.slice_ndarray(x_in, idx=4, length=2, axis=2),
        x_in[:, :, 4:6])


def test_padded_slice_ndarray():
    x_in = np.arange(64).reshape(2, 4, 8)
    np.testing.assert_equal(
        utils.padded_slice_ndarray(x_in, idx=0, length=4, axis=0),
        np.concatenate([np.zeros([2, 4, 8]), x_in[:2, ...]], axis=0))
