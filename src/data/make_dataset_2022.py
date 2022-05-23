from pathlib import Path
import logging

logging.basicConfig(
    filename="dicom_conversion_chup_v2.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import click
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter
from okapy.dicomconverter.dicom_walker import DicomWalker

from src.data.utils import (correct_names, sort_vois, combine_vois, clean_vois,
                            compute_bbs)

project_dir = Path(__file__).resolve().parents[2]
# default_input_path = project_dir / "hecktor/data/hecktor2022/raw/mda_test"
center = "CHUP_v2"
# default_input_path = f"/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2022/dicom/{center}"
default_input_path = f"/media/val/Windows/Users/valen/Documents/work/{center}/"
default_images_folder = project_dir / f"data/hecktor2022/processed/{center}/images"
default_labels_original_folder = project_dir / f"data/hecktor2022/processed/{center}/labels_original"
default_labels_folder = project_dir / f"data/hecktor2022/processed/{center}/labels"
default_dump = project_dir / f"data/hecktor2022/processed/{center}/dump"
default_name_mapping = project_dir / "data/hecktor2021_name_mapping_testing.csv"


def filter_func(study):
    # return study.patient_id == "CHB003"

    error_list = [
        'HN-HGJ-010',
    ]
    return study.patient_id in error_list


@click.command()
@click.argument('input_folder', type=click.Path(), default=default_input_path)
@click.argument('output_images_folder',
                type=click.Path(),
                default=default_images_folder)
@click.argument('output_labels_folder',
                type=click.Path(),
                default=default_labels_folder)
@click.argument('output_labels_original_folder',
                type=click.Path(),
                default=default_labels_original_folder)
@click.argument('dump_folder', type=click.Path(), default=default_dump)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
def main(input_folder, output_images_folder, output_labels_folder,
         output_labels_original_folder, dump_folder, name_mapping):
    """Command Line Interface to make the dataset for the HECKTOR Challenge
        In short, this routine convert the DICOM files to NIFTI and stores the CT in 
        Hounsfield Unit and the PET in Standardized Uptake value.
    """

    output_images_folder = Path(output_images_folder)
    output_labels_folder = Path(output_labels_folder)
    output_labels_original_folder = Path(output_labels_original_folder)
    dump_folder = Path(dump_folder)
    output_images_folder.mkdir(exist_ok=True, parents=True)
    output_labels_folder.mkdir(exist_ok=True, parents=True)
    output_labels_original_folder.mkdir(exist_ok=True, parents=True)
    dump_folder.mkdir(exist_ok=True, parents=True)
    # logger.info("Converting Dicom to Nifty - START")
    # converter = NiftiConverter(
    #     padding="whole_image",
    #     labels_startswith="GTV",
    #     # dicom_walker=DicomWalkerWithFilter(filter_func=filter_func, cores=24),
    #     dicom_walker=DicomWalker(cores=12),
    #     # cores=12,
    #     cores=None,
    #     naming=2,
    # )
    # conversion_results = converter(input_folder,
    #                                output_folder=output_images_folder)
    # list_errors = [
    #     d.get("patient_id") for d in conversion_results
    #     if d.get("status") == "failed"
    # ]
    # print(f"List of patients with errors: {list_errors}")
    # logger.info("Converting Dicom to Nifty - END")
    # logger.info("Removing extra VOI - START")
    # sort_vois(output_images_folder, output_labels_original_folder, dump_folder)
    # logger.info("Removing extra VOI - END")
    logger.info("Combining all VOIs into one file - START")
    combine_vois(output_labels_original_folder, output_labels_folder)
    logger.info("Combining all VOIs into one file - END")
    # logger.info("Renaming files- START")
    # correct_names(output_images_folder, name_mapping)
    # logger.info("Renaming files- END")
    # logger.info("Cleaning the VOIs - START")
    # clean_vois(output_images_folder)
    # logger.info("Cleaning the VOIs - END")


class DicomWalkerWithFilter(DicomWalker):

    def __init__(self,
                 input_dirpath=None,
                 cores=None,
                 additional_dicom_tags=None,
                 submodalities=False,
                 filter_func=None):
        super().__init__(input_dirpath, cores, additional_dicom_tags,
                         submodalities)
        self.filter_func = filter_func

    def __call__(self, input_dirpath=None, cores=None):
        studies = super().__call__(input_dirpath, cores)
        return [s for s in studies if self.filter_func(s)]


if __name__ == '__main__':
    main()
