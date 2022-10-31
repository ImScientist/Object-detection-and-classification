import os
import click
import logging
import warnings

import src.settings as settings
from src.data.preprocessing import create_tr_va_te_datasets

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning)

data_raw_dir = os.path.join(settings.DATA_DIR, 'raw')
data_preprocessed_dir = os.path.join(settings.DATA_DIR, 'preprocessed')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--tr', default=.8, type=float, help='training fraction')
@click.option('--va', default=.1, type=float, help='validation fraction')
@click.option('--te', default=.1, type=float, help='test fraction')
@click.option('--rf', default=1, type=int, help='reduction factor of image dimensions')
def create_datasets(tr, va, te, rf):
    """ Preprocess data and split it into a training, validation and test
    datasets """

    assert tr + va + te == 1

    create_tr_va_te_datasets(
        source_dir=os.path.join(settings.DATA_DIR, 'raw'),
        output_dir=os.path.join(settings.DATA_DIR, f'preprocessed_rf_{rf}'),
        tr_va_te_frac=(tr, va, te),
        reduction_factor=rf)


if __name__ == "__main__":
    """
        PYTHONPATH=$(pwd) python src/main.py --help    
        PYTHONPATH=$(pwd) python src/main.py create-datasets --rf=4
    """

    cli()
