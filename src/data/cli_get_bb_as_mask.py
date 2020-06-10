import os
from pathlib import Path
import logging

import click
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.resampling.utils import (get_np_volume_from_sitk,
                                  get_sitk_volume_from_np)

# Default paths
path_in = 'data/hecktor_nii/'
path_out = 'data/bbox_nii/'
path_bb = 'data/bbox.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
def main(input_folder, output_folder, bounding_boxes_file):
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
    logger.info('Starting to write the bb to NIFTI')

    bb_dict = pd.read_csv(bounding_boxes_file).set_index('PatientID')
    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetImageIO('NiftiImageIO')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for f in Path(input_folder).rglob('*_ct.nii.gz'):
        patient_name = f.name.split('_')[0]
        np_volume, spacing, origin = get_np_volume_from_sitk(
            sitk.ReadImage(str(f.resolve())))
        bb = np.round((np.asarray([
            bb_dict.loc[patient_name, 'x1'],
            bb_dict.loc[patient_name, 'y1'],
            bb_dict.loc[patient_name, 'z1'],
            bb_dict.loc[patient_name, 'x2'],
            bb_dict.loc[patient_name, 'y2'],
            bb_dict.loc[patient_name, 'z2'],
        ]) - np.tile(origin, 2)) / np.tile(spacing, 2)).astype(int)
        np_mask_bb = np.zeros_like(np_volume).astype(np.uint8)
        np_mask_bb[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]] = 1
        sitk_mask_bb = get_sitk_volume_from_np(np_mask_bb, spacing, origin)
        output_filepath = os.path.join(output_folder,
                                       patient_name + "_bbox.nii.gz")
        sitk_writer.SetFileName(output_filepath)
        sitk_writer.Execute(sitk_mask_bb)
        logger.info('{} done'.format(patient_name))
    logger.info('All done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
