from pathlib import Path
import logging

logging.basicConfig(
    filename="dicom_conversion.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import click
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter

from src.data.utils import (correct_names, move_extra_vois, clean_vois,
                            compute_bbs)

project_dir = Path(__file__).resolve().parents[2]
default_input_path = project_dir / "data/raw"
default_images_folder = project_dir / "data/hecktor2021/hecktor_nii/"
default_files_folder = project_dir / "data/hecktor2021/"
default_archive = project_dir / "data/extra_voi"
default_name_mapping = project_dir / "data/hecktor2021_name_mapping_testing.csv"
default_bb_file = project_dir / "data/hecktor2021/hecktor2021_bbox_training.csv"


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
@click.argument('bb_file', type=click.Path(), default=default_bb_file)
@click.option("--archive_folder",
              "-a",
              type=click.Path(),
              default=default_archive)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
def main(input_folder, output_images_folder, output_files_folder, bb_file,
         archive_folder, name_mapping):
    """Command Line Interface to make the dataset for the HECKTOR Challenge
        In short, this routine convert the DICOM files to NIFTI and stores the CT in 
        Hounsfield Unit and the PET in Standardized Uptake value.
    """

    output_images_folder = Path(output_images_folder)
    output_files_folder = Path(output_files_folder)
    archive_folder = Path(archive_folder)
    output_images_folder.mkdir(exist_ok=True, parents=True)
    archive_folder.mkdir(exist_ok=True, parents=True)
    # logger.info("Converting Dicom to Nifty - START")
    # converter = NiftiConverter(
    #     padding="whole_image",
    #     list_labels=["GTVt", "GTVn"],
    #     cores=10,
    #     naming=2,
    # )
    # _, _ = converter(input_folder, output_folder=output_images_folder)

    logger.info("Converting Dicom to Nifty - END")
    # logger.info("Removing extra VOI - START")
    # move_extra_vois(output_images_folder, archive_folder)
    # logger.info("Removing extra VOI - END")
    # logger.info("Renaming files- START")
    # correct_names(output_images_folder, name_mapping)
    # logger.info("Renaming files- END")
    # logger.info("Cleaning the VOIs - START")
    clean_vois(output_images_folder)
    logger.info("Cleaning the VOIs - END")

    # logger.info("Computing the bounding boxes - START")
    # bb_df = compute_bbs(output_images_folder)
    # bb_df.to_csv(bb_file)
    # logger.info("Computing the bounding boxes - END")


if __name__ == '__main__':
    main()
