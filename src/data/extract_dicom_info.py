from pathlib import Path
import json
import logging

logging.basicConfig(
    filename="dicom_info_extraction.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

import click
import pandas as pd
from okapy.dicomconverter.dicom_walker import DicomWalker

from src.data.utils import map_names_hecktor

project_dir = Path(__file__).resolve().parents[2]
# data_dir = Path("/run/media/val/083C23E228226C35/work/hecktor2022/processed/")
data_dir = project_dir / "data/hecktor2022/"


def get_pure_center_name(center):
    if "mda_test" in center:
        return "mda_test"
    return center.split("_")[0].lower()


with open(data_dir / "paths_to_dicom.json", "r") as f:
    center_paths = json.load(f)

# default_input_path = project_dir / "hecktor/data/hecktor2022/raw/mda_test"
center_folder = "mda_test_corrected_v3"
center = get_pure_center_name(center_folder)
default_input_path = center_paths[center_folder]
default_name_mapping = project_dir / f"data/hecktor2022/mappings/name_mapping_{center}.csv"
default_output_filepath = project_dir / f"data/hecktor2022/dicom_info_{center}.csv"


@click.command()
@click.argument('input_folder', type=click.Path(), default=default_input_path)
@click.argument('output_filepath',
                type=click.Path(),
                default=default_output_filepath)
@click.option("--name_mapping",
              type=click.Path(),
              default=default_name_mapping)
def main(input_folder, output_filepath, name_mapping):
    walker = DicomWalker(additional_dicom_tags="PatientWeight")
    studies = walker(input_dirpath=input_folder, cores=12)
    output = pd.DataFrame()
    for study in studies:

        tmp_df = pd.DataFrame()
        tmp_df["PatientID"] = [study.patient_id]
        tmp_df["StudyDate"] = [study.study_date]
        try:
            pt_volume = [v for v in study.volume_files
                         if v.modality == "PT"][0]
        except:
            logger.warning(
                f"No PT volume found for {study.patient_id}, "
                f"maybe PT is an other study? (study instance UID: {study.study_instance_uid})"
            )
            continue
        try:
            tmp_df["patient_weight"] = [pt_volume.patient_weight]
        except:
            logger.error(f"{study.patient_id} has no weight")
            tmp_df["patient_weight"] = [-1]
        output = pd.concat([output, tmp_df], ignore_index=True)

    output["hecktor_id"] = output["PatientID"].map(
        lambda x: map_names_hecktor(x, name_mapping, center=center))

    output.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    main()
