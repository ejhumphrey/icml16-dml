import argparse
import os
import yaml

import dml.data
import dml.driver
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
    config['train'] = dml.driver.fit(kwargs, config_out)
    return config


def validate(config):
    pass


def predict(config):
    pass


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
