import argparse
import numpy as np
import optimus
import os
import pandas as pd
import yaml

import dml.data
import dml.driver
import dml.models
import dml.utils


def build(config):
    kwargs = config['build']

    dataset = dml.data.index_directory(kwargs['data_root'])

    train, test = dml.data.split_dataset(
        dataset, kwargs['train_test_ratio'])
    train, valid = dml.data.split_dataset(
        train, kwargs['train_valid_ratio'])

    for k, dset in {'train': train, 'valid': valid, 'test': test}.items():
        fout = kwargs['{}_file'.format(k)]
        dml.utils.safe_makedirname(fout)
        dset.to_json(fout)

        if not os.path.exists(fout):
            raise ValueError(
                "Failed to write data partition {}: {}".format(k, fout))
    return config


def train(config):
    kwargs = config['train']
    config_out = "{trial_name}_config.yaml".format(**kwargs)
    config['train'] = dml.driver.main(kwargs, config_out)
    return config


def validate(config):
    pass


def predict(config):
    kwargs = config['predict']
    graph = optimus.load(kwargs['model_file'])
    # TODO: Optimus bug, fix this.
    graph.param_values = dict(**np.load(kwargs['param_file']))
    dataset = pd.read_json(kwargs['dataset'])
    new_dataset = dml.models.transform_dataset(
        dataset, graph, kwargs['output_dir'])
    output_file = os.path.join(kwargs['output_dir'], "output_index.json")
    new_dataset.to_json(output_file)
    return new_dataset


def score(config):
    pass


def main(config):
    pipeline = [build, train, validate, predict, score]
    for fx in pipeline:
        config = fx(config)


STAGES = {
    'build': build,
    'train': train,
    'validate': validate,
    'predict': predict,
    'score': score,
    'all': main,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path to a yaml config file.")
    parser.add_argument("stage",
                        metavar="stage", type=str,
                        help="Stage to execute.")
    args = parser.parse_args()
    config = yaml.load(open(args.config_file))
    STAGES.get(args.stage)(config)
