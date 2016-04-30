"""Primary training driver for the NLSE models.

Example
-------
$ python dml/driver.py \
    data/nlse_config.yaml \
    --output_file training_config.yaml
"""

from __future__ import print_function

import argparse
import optimus
import os
import pandas as pd
import yaml

import dml.data as D
import dml.models as M
import dml.utils as utils


def fit(trial_name, output_dir, model_params, hyperparams,
        train_params, data_params, param_file=''):
    """Fit a model given the parameters.

    Parameters
    ----------
    ...

    Returns
    -------
    artifacts : dict
        Contains data resulting from fitting the model.
    """
    output_dir = os.path.join(output_dir, trial_name)
    utils.safe_makedirs(output_dir)

    trainer, predictor = M.create(**model_params)

    data_params.update(
        window_length=model_params['n_in'],
        dataset=pd.read_json(data_params.pop('dataset')))

    # TODO: Migrate this into the config object... but where?!
    source = D.awgn(D.create_stream(**data_params), 0.1, 0.01)

    print("Starting '{0}'".format(trial_name))
    param_file = os.path.join(output_dir, param_file)

    log_file = os.path.join(output_dir, 'train_stats.csv')
    driver = optimus.Driver(
        graph=trainer,
        name=trial_name,
        output_directory=output_dir,
        log_file=log_file)

    model_file = os.path.join(output_dir,
                              "{}-predictor.json".format(trial_name))
    optimus.save(predictor, model_file)
    driver.fit(source, hyperparams=hyperparams, **train_params)
    return dict(log_file=log_file, model_file=model_file)


def main(config, filename=''):
    """Run the main driver.

    Parameters
    ----------
    config : dict
        State object for training.

    filename : str
        Filename for writing the updated config file.

    Returns
    -------
    output_config : dict
        Updated config file.
    """
    output_dir = config['output_dir']

    config.update(model_outputs=fit(**config.copy()))

    # Snapshot this artifact.
    if filename:
        output_config = os.path.join(output_dir, filename)
        try:
            with open(output_config, 'w') as fp:
                yaml.dump(config, fp)
        except TypeError as derp:
            print("Saving config failed: {}\n\n{}"
                  "".format(config, derp))

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Inputs
    parser.add_argument("config_file",
                        metavar="config_file", type=str,
                        help="Path to a yaml config file.")
    parser.add_argument("--output_file",
                        metavar="output_file", type=str, default='',
                        help="Path to a yaml config file.")
    args = parser.parse_args()
    config = yaml.load(open(args.config_file))
    main(config, args.output_file)
