from pathlib import Path
import logging

import click
import SimpleITK as sitk
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter

from src.data.bounding_box import bbox_auto
from src.data.utils import keep_newest_rtstruct, correct_filename

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).resolve().parents[2]
default_input_path = project_dir / "data/raw"
default_images_folder = project_dir / "data/hecktor2021/hecktor_nii/"
default_files_folder = project_dir / "data/hecktor2021/"


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default=default_input_path)
@click.argument('output_images_folder',
                type=click.Path(),
                default=default_images_folder)
@click.argument('output_files_folder',
                type=click.Path(),
                default=default_files_folder)
@click.option('--subfolders', is_flag=True)
def main(input_folder, output_images_folder, output_files_folder, subfolders):
    """Command Line Interface to make the conversion from DICOM to NIFTI,
       computing Hounsfield units and SUV for CT and PT respectively.

    Args:
        input_folder (str): Path to the folder containing the DICOM files
        output_folder_nii (str): Path to the folder where the NIFTI will be
                               stored.
        subfolders (bool): Wether to create one subfolder for each patient
                           or to store everything within the same directory.

    Raises:
        ValueError: Raised if more than 1 RTSTRUCT file is found
    """
    logger.info("Keeping only newest RTSTRUCT per patient")
    keep_newest_rtstruct("/home/val/python_wkspce/hecktor/data/raw/poitier_train/",
                         project_dir / "data/surnumerary_rtstructs/")
    logger.info("Converting Dicom to Nifty - START")

    # converter = NiftiConverter(
    #     padding="whole_image",
    #     resampling_spacing=-1,
    #     list_labels=["GTVt"],
    #     cores=10,
    # )
    # _ = converter(input_folder, output_folder=output_images_folder)

    # logger.info("Converting Dicom to Nifty - END")

    # bb = bbox_auto(np_pt, px_spacing_pt, im_pos_patient_pt)
    # bb_df = bb_df.append(
    #     {
    #         'PatientID': patient_name,
    #         'x1': bb[0],
    #         'x2': bb[1],
    #         'y1': bb[2],
    #         'y2': bb[3],
    #         'z1': bb[4],
    #         'z2': bb[5],
    #     },
    #     ignore_index=True)
    # bb_df.to_csv(bb_filepath)


if __name__ == '__main__':
    main()
