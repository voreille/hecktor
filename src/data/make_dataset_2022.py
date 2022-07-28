from pathlib import Path
import json
import logging
from shutil import move

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import click
import numpy as np
import pandas as pd
from okapy.dicomconverter.converter import NiftiConverter
from okapy.dicomconverter.dicom_walker import DicomWalker

from src.data.utils import (correct_names, sort_vois, combine_vois, clean_vois,
                            correct_images_direction, compute_bbs)

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data/hecktor2022/"

# Can have multiple folder for the same center depending on the version, with corrections or else
# The usual pattern I use is <center_name>_[corrected_]v<version> with [corrected_] being optional
# see the example in the the paths_to_dicom.json

center_folder = "mda_test"

fh = logging.FileHandler(f"dicom_conversion_{center_folder}.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_pure_center_name(center):
    if "mda_test" in center:
        return "mda_test"
    return center.split("_")[0].lower()


with open(data_dir / "paths_to_dicom.json", "r") as f:
    center_paths = json.load(f)

default_input_path = center_paths[center_folder]
default_images_folder = data_dir / f"processed/{center_folder}/images"
default_labels_original_folder = data_dir / f"processed/{center_folder}/labels_original"
default_labels_folder = data_dir / f"processed/{center_folder}/labels"
default_dump = data_dir / f"processed/{center_folder}/dump"
default_name_mapping = data_dir / f"mappings/name_mapping_{get_pure_center_name(center_folder)}.csv"
voi_mapping = data_dir / f"mappings/vois_mapping_{get_pure_center_name(center_folder)}.json"

if voi_mapping.exists():
    with open(voi_mapping, "r") as f:
        voi_mapping = json.load(f)
else:
    voi_mapping = None


def filter_func(study):
    # return study.patient_id == "CHB003"

    error_list = [
        "2827563172",
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
@click.option("--clean-nifty",
              is_flag=True,
              show_default=True,
              default=False,
              help="Clean the nifty files")
@click.option("--clean-all",
              is_flag=True,
              show_default=True,
              default=False,
              help="Remove all the nifty files")
def main(input_folder, output_images_folder, output_labels_folder,
         output_labels_original_folder, dump_folder, name_mapping, clean_nifty,
         clean_all):
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

    image_renamed_folder = output_images_folder.parent / "images_renamed"
    image_renamed_folder.mkdir(exist_ok=True, parents=False)

    label_renamed_folder = output_images_folder.parent / "labels_renamed"
    label_renamed_folder.mkdir(exist_ok=True, parents=False)

    if clean_nifty:
        for f in dump_folder.glob("*.nii.gz"):
            move(f, output_images_folder)
        for f in output_labels_original_folder.glob("*.nii.gz"):
            move(f, output_images_folder)
        for f in output_labels_folder.glob("*.nii.gz"):
            move(f, output_images_folder)
        for f in image_renamed_folder.glob("*.nii.gz"):
            f.unlink()
        for f in label_renamed_folder.glob("*.nii.gz"):
            f.unlink()
        return

    if clean_all:
        for f in output_images_folder.glob("*.nii.gz"):
            f.unlink()
        for f in dump_folder.glob("*.nii.gz"):
            f.unlink()
        for f in output_labels_folder.glob("*.nii.gz"):
            f.unlink()
        for f in output_labels_original_folder.glob("*.nii.gz"):
            f.unlink()
        for f in image_renamed_folder.glob("*.nii.gz"):
            f.unlink()
        for f in label_renamed_folder.glob("*.nii.gz"):
            f.unlink()
        return

    # logger.info("Converting Dicom to Nifty - START")
    # if "montreal" in center_folder.lower():
    #     labels_startswith = "GTV"
    # else:
    #     labels_startswith = None

    # converter = NiftiConverter(
    #     padding="whole_image",
    #     labels_startswith=labels_startswith,
    #     # dicom_walker=DicomWalkerWithFilter(filter_func=filter_func, cores=12),
    #     dicom_walker=DicomWalker(cores=12),
    #     cores=12,
    #     # cores=None,
    #     naming=2,
    # )
    # conversion_results = converter(input_folder,
    #                                output_folder=output_images_folder)
    # list_errors = [
    #     d.get("patient_id") for d in conversion_results
    #     if d.get("status") == "failed"
    # ]
    # logger.info(f"List of patients with errors: {list_errors}")
    # logger.info("Converting Dicom to Nifty - END")
    logger.info("Removing extra VOI - START")
    sort_vois(output_images_folder,
              output_labels_original_folder,
              dump_folder,
              center=center_folder,
              voi_mapping=voi_mapping)
    logger.info("Removing extra VOI - END")
    logger.info("Combining all VOIs into one file - START")
    combine_vois(output_labels_original_folder,
                 output_labels_folder,
                 dump_folder,
                 center=center_folder,
                 voi_mapping=voi_mapping)
    logger.info("Combining all VOIs into one file - END")
    logger.info("Renaming files- START")
    correct_names(output_images_folder,
                  image_renamed_folder,
                  name_mapping,
                  center=center_folder)
    correct_names(output_labels_folder,
                  label_renamed_folder,
                  name_mapping,
                  center=center_folder)
    logger.info("Renaming files- END")
    logger.info("Cleaning the VOIs - START")
    clean_vois(label_renamed_folder)
    logger.info("Cleaning the VOIs - END")
    logger.info("Cleaning the VOIs - START")
    correct_images_direction(image_renamed_folder, label_renamed_folder)
    logger.info("Cleaning the VOIs - END")


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
