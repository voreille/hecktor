from pathlib import Path
import json
import logging
from shutil import move

from traitlets import default

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import click
from okapy.dicomconverter.converter import NiftiConverter
from okapy.dicomconverter.dicom_walker import DicomWalker

from src.data.utils import (correct_names_with_mapping, sort_vois,
                            combine_vois, clean_vois, correct_images_direction,
                            crop_images)

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data/interobserver"

# Can have multiple folder for the same center depending on the version, with corrections or else
# The usual pattern I use is <center_name>_[corrected_]v<version> with [corrected_] being optional
# see the example in the the paths_to_dicom.json

fh = logging.FileHandler(f"dicom_conversion_interobserver.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

default_input_path = data_dir / "dicom"
default_output_path = data_dir / "nii"
default_name_mapping = project_dir / "data/hecktor2022/mappings/name_mapping_total.csv"
bb_file = data_dir / "data/hecktor2022/mappings/bb_mapping.json"


def filter_func(study):
    # return study.patient_id == "CHB003"

    error_list = [
        "CHB039",
    ]
    return study.patient_id in error_list


@click.command()
@click.argument('input_folder', type=click.Path(), default=default_input_path)
@click.argument('output_folder',
                type=click.Path(),
                default=default_output_path)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
def main(input_folder, output_folder, name_mapping):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    logger.info("Converting Dicom to Nifty - START")
    converter = NiftiConverter(
        padding="whole_image",
        # dicom_walker=DicomWalkerWithFilter(filter_func=filter_func, cores=12),
        dicom_walker=DicomWalker(cores=12),
        cores=12,
        # cores=None,
        naming=2,
    )
    for interobserver_dir in input_folder.iterdir():
        observer = interobserver_dir.name
        voi_mapping = project_dir / f"data/interobserver/vois_mapping_{observer}.json"
        with open(voi_mapping, "r") as f:
            voi_mapping = json.load(f)

        logger.info(f"Processing {observer}")
        images_folder = output_folder / f"{observer}/images"
        images_renamed_folder = output_folder / f"{observer}/images_renamed"
        labels_renamed_folder = output_folder / f"{observer}/labels_renamed"
        labels_original_folder = output_folder / f"{observer}/labels_original"
        labels_folder = output_folder / f"{observer}/labels"
        dump_folder = output_folder / f"{observer}/dump"

        images_folder.mkdir(exist_ok=True, parents=True)
        labels_folder.mkdir(exist_ok=True, parents=True)
        labels_original_folder.mkdir(exist_ok=True, parents=True)
        dump_folder.mkdir(exist_ok=True, parents=True)
        images_renamed_folder.mkdir(exist_ok=True, parents=True)
        labels_renamed_folder.mkdir(exist_ok=True, parents=True)

        conversion_results = converter(interobserver_dir,
                                       output_folder=images_folder)
        list_errors = [
            d.get("patient_id") for d in conversion_results
            if d.get("status") == "failed"
        ]
        logger.info(f"List of patients with errors: {list_errors}")
        logger.info("Converting Dicom to Nifty - END")
        logger.info("Removing extra VOI - START")
        sort_vois(images_folder,
                  labels_original_folder,
                  dump_folder,
                  voi_mapping=voi_mapping,
                  keep_only_latest_rtstruct=True)
        logger.info("Removing extra VOI - END")
        logger.info("Combining all VOIs into one file - START")
        combine_vois(labels_original_folder,
                     labels_folder,
                     dump_folder,
                     voi_mapping=voi_mapping)
        logger.info("Combining all VOIs into one file - END")
        logger.info("Renaming files- START")
        correct_names_with_mapping(images_folder, images_renamed_folder,
                                   name_mapping)
        correct_names_with_mapping(labels_folder, labels_renamed_folder,
                                   name_mapping)
        logger.info("Renaming files- END")
        logger.info("Cleaning the VOIs - START")
        clean_vois(labels_renamed_folder)
        logger.info("Cleaning the VOIs - END")
        logger.info("Cleaning the VOIs - START")
        correct_images_direction(images_renamed_folder, labels_renamed_folder)
        logger.info("Cleaning the VOIs - END")
        if bb_file.exists():
            logger.info("Cropping Images - START")
            crop_images(images_renamed_folder, images_renamed_folder, bb_file)
            crop_images(labels_renamed_folder, labels_renamed_folder, bb_file)
            logger.info("Cropping Images - END")


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
