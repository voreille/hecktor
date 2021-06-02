from pathlib import Path
import logging

import click
import SimpleITK as sitk
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter

from src.data.bounding_box import bbox_auto
from src.data.utils import (correct_names, move_extra_vois, clean_vois,
                            compute_bbs)

project_dir = Path(__file__).resolve().parents[2]
default_input_path = project_dir / "data/raw"
default_images_folder = project_dir / "data/hecktor2021/hecktor_nii/"
default_files_folder = project_dir / "data/hecktor2021/"
default_archive = project_dir / "data/surnumerary_voi"
default_name_mapping = project_dir / "data/hecktor2021/name_mapping_training.csv"

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=project_dir / "data/dataset_processing.log",
                    encoding='utf-8',
                    level=logging.INFO,
                    format=log_fmt)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


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
@click.option("--archive_folder",
              "-a",
              type=click.Path(),
              default=default_archive)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
@click.option('--subfolders', is_flag=True)
def main(input_folder, output_images_folder, output_files_folder,
         archive_folder, name_mapping, subfolders):
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
    output_images_folder = Path(output_images_folder)
    output_files_folder = Path(output_files_folder)
    archive_folder = Path(archive_folder)
    output_images_folder.mkdir(exist_ok=True)
    archive_folder.mkdir(exist_ok=True)
    # logger.info("Converting Dicom to Nifty - START")
    # converter = NiftiConverter(
    #     padding="whole_image",
    #     resampling_spacing=-1,
    #     list_labels=["GTVt"],
    #     cores=10,
    # )
    # _ = converter(input_folder, output_folder=output_images_folder)

    # logger.info("Converting Dicom to Nifty - END")
    # logger.info("Removing extra VOI - START")
    # move_extra_vois(output_images_folder, archive_folder)
    # logger.info("Removing extra VOI - END")
    # logger.info("Renaming files- START")
    # correct_names(output_images_folder, name_mapping)
    # logger.info("Renaming files- END")
    logger.info("Cleaning the VOIs - START")
    clean_vois(output_images_folder)
    logger.info("Cleaning the VOIs - END")

    logger.info("Computing the bounding boxes - START")
    bb_df = compute_bbs(output_images_folder)
    bb_df.to_csv(output_files_folder / "bounding_boxes_training.csv")
    logger.info("Computing the bounding boxes - END")


if __name__ == '__main__':
    main()
