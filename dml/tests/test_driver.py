import pytest

import glob
import os
import yaml

import dml.data as data
import dml.driver as driver


@pytest.fixture()
def config_file(test_dir):
    return os.path.join(test_dir, 'simple_config.yaml')


def test_config_exists(config_file):
    assert os.path.exists(config_file)


@pytest.fixture()
def dataset(data_dir):
    return data.index_directory(data_dir)


def test_driver_instrument(config_file, dataset, workspace):
    index_file = os.path.join(workspace, "data_index.json")
    dataset.to_json(index_file)
    config = yaml.load(open(config_file))
    config['output_dir'] = workspace
    config['data_params']['dataset'] = index_file
    config['data_params']['neighbor_mode'] = 'instrument'
    driver.fit(**config)

    assert glob.glob(os.path.join(workspace, 'train_stats.*'))
    assert glob.glob(os.path.join(workspace, '*-predictor.json'))
    assert glob.glob(os.path.join(workspace, '*.npz'))


def test_driver_pitch(config_file, dataset, workspace):
    index_file = os.path.join(workspace, "data_index.json")
    dataset.to_json(index_file)
    config = yaml.load(open(config_file))
    config['output_dir'] = workspace
    config['data_params']['dataset'] = index_file
    config['data_params']['neighbor_mode'] = 'pitch'
    driver.fit(**config)

    assert glob.glob(os.path.join(workspace, 'train_stats.*'))
    assert glob.glob(os.path.join(workspace, '*-predictor.json'))
    assert glob.glob(os.path.join(workspace, '*.npz'))


def test_driver_instpitch(config_file, dataset, workspace):
    index_file = os.path.join(workspace, "data_index.json")
    dataset.to_json(index_file)
    config = yaml.load(open(config_file))
    config['output_dir'] = workspace
    config['data_params']['dataset'] = index_file
    config['data_params']['neighbor_mode'] = 'instrument-pitch'
    driver.fit(**config)

    assert glob.glob(os.path.join(workspace, 'train_stats.*'))
    assert glob.glob(os.path.join(workspace, '*-predictor.json'))
    assert glob.glob(os.path.join(workspace, '*.npz'))
