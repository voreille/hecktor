import os
from pathlib import Path

import click
import logging
import pandas as pd
import SimpleITK as sitk

from src.resampling.utils import (get_np_volume_from_sitk,
                                  get_sitk_volume_from_np)

# Default paths
path_in = 'data/processed'
path_out = 'data/bbox_nii/'
path_bb = 'data/bbox.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
def main(input_folder, output_folder, bounding_boxes_file, cores, resampling,
         order):
    """ This command line interface allows to obtain the bounding boxes
        contained in BOUNDING_BOXES_FILE as a NIFTI mask. The NIFTI files are
        stored in OUTPUT_FOLDER and they contain the value 0 and 1 for outside
        or inside the bounding boxe resepectively.

        INPUT_FOLDER is the path of the folder containing the NIFTI
        in the original reference frame and resolution (the one downloaded
        from AIcrowd):
        OUTPUT_FOLDER is the path of the folder where to store the
        NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    ct_files = [f for f in Path(input_folder).rglob('*_ct.nii.gz')]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
