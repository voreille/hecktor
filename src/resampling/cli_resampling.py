import os
from multiprocessing import Pool
import glob

import click
import logging
import pandas as pd

from src.resampling.resampling import Resampler

# Default paths
path_in = 'data/hecktor_nii/'
path_out = 'data/resampled/'
path_bb = 'data/bbox.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1),
              help='Expect 3 positive floats describing the output '
              'resolution of the resampling. To avoid resampling '
              'on one or more dimension a value of -1 can be fed '
              'e.g. --resampling 1.0 1.0 -1 will resample the x '
              'and y axis at 1 mm/px and left the z axis untouched.')
@click.option('--order',
              type=click.INT,
              nargs=1,
              default=3,
              help='The order of the spline interpolation used to resample')
def main(input_folder, output_folder, bounding_boxes_file, cores, resampling,
         order):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print('resampling is {}'.format(str(resampling)))
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    files_list = [
        f for f in glob.glob(input_folder + '/**/*.nii.gz', recursive=True)
    ]
    resampler = Resampler(bb_df, output_folder, order, resampling=resampling)
    with Pool(cores) as p:
        p.map(resampler, files_list)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
