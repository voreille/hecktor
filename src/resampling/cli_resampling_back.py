import os
from multiprocessing import Pool
from pathlib import Path

import click
import logging
import pandas as pd

from src.resampling.resampling import Resampler

path_input = "segmentation_output_renamed"
path_output = "data/segmentation_output_tosubmit"
path_bb = "data/bbox.csv"
path_res = "data/original_resolution_ct.csv"


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default=path_input)
@click.argument('output_folder', type=click.Path(), default=path_output)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.argument('original_resolution_file',
                type=click.Path(),
                default=path_res)
@click.option('--cores', type=click.INT, default=1)
def main(input_folder, output_folder, bounding_boxes_file,
         original_resolution_file, cores):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    resolution_df = pd.read_csv(original_resolution_file)
    resolution_df = resolution_df.set_index('PatientID')
    files_list = [
        str(f.resolve()) for f in Path(input_folder).rglob('*.nii.gz')
    ]
    patient_list = [
        f.name[:7] for f in Path(input_folder).rglob('*.nii.gz')
    ]
    resampler = Resampler(bb_df, output_folder, order='nearest')
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
