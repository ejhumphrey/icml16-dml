"""Primary training driver for the NLSE models.

Example
-------
$ python scripts/training_driver.py \
    data/nlse_config.yaml \
    output_dir \
    --verbose
"""

from __future__ import print_function
import argparse
import optimus
import pandas as pd
import os
import shutil
import yaml

import dml.data as D
import dml.models as M
import dml.utils as utils


def fit(trial_name, output_dir, model_params, hyperparams,
        train_params, data_params, param_file=''):

    trainer, predictor = M.create(**model_params)

    data_params.update(
        window_length=model_params['n_in'],
        dataset=pd.read_json(data_params.pop('dataset')))

    source = D.awgn(D.create_stream(**data_params), 0.1, 0.01)

    print("Starting '{0}'".format(trial_name))
    param_file = os.path.join(output_dir, param_file)

    log_file = os.path.join(output_dir, 'train_stats.csv')
    driver = optimus.Driver(
        graph=trainer,
        name=trial_name,
        output_directory=output_dir,
        log_file=log_file,
        display=False)

    model_file = os.path.join(output_dir,
                              "{}-predictor.json".format(trial_name))
    optimus.save(predictor, model_file)
    driver.fit(source, hyperparams=hyperparams, **train_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path to a yaml config file.")
    args = parser.parse_args()
    kwargs = yaml.load(open(args.config_file))

    output_dir = kwargs['output_dir']
    utils.safe_makedirs(output_dir)
    shutil.copy(args.config_file, output_dir)
    fit(**kwargs)
