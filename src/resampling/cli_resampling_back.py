import os
from multiprocessing import Pool
import glob

import click
import logging
import pandas as pd

from src.resampling.resampling import Resampler


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default='data/processed')
@click.argument('output_folder', type=click.Path(), default='data/results')
@click.argument('bounding_boxes_file',
                type=click.Path(),
                default='data/bb.csv')
@click.argument('original_resolution_file',
                type=click.Path(),
                default='data/original_resolution.csv')
@click.option('--cores',
              type=click.INT,
              default=1,
              help='The number of workers for parallelization.')
@click.option('--order',
              type=click.INT,
              nargs=1,
              default=3,
              help='The order of the spline interpolation used to resample')
def main(input_folder, output_folder, bounding_boxes_file,
         original_resolution_file, cores, order):
    """ This command line interface allows to resample NIFTI files back to the
        original resolution contained in ORIGINAL_RESOLUTION_FILE (this file
        can be gerenated with the file src/resampling/cli_get_resolution.py).
        It also needs the bounding boxes contained in BOUNDING_BOXES_FILE.
        The images are resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled by
        nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    resolution_df = pd.read_csv(original_resolution_file)
    resolution_df = resolution_df.set_index('PatientID')
    files_list = [
        f for f in glob.glob(input_folder + '/**/*_GTV?.nii', recursive=True)
    ]
    resampler = Resampler(bb_df, output_folder, order)
    patient_list = [f.split('/')[-2] for f in files_list]
    resolution_list = [(resolution_df.loc[k, 'Resolution_x'],
                        resolution_df.loc[k, 'Resolution_y'],
                        resolution_df.loc[k, 'Resolution_z'])
                       for k in patient_list]
    with Pool(cores) as p:
        p.starmap(resampler, zip(files_list, resolution_list))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
