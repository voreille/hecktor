from pathlib import Path
import json
import logging
from shutil import copy

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import pandas as pd
from okapy.dicomconverter.dicom_walker import DicomWalker

project_dir = Path(__file__).resolve().parents[2]

fh = logging.FileHandler(f"sort_cases_per_annotator.log")
fh.setLevel(logging.WARNING)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def main(input_folders, output_folder, list_cases_file, name_mapping):
    mapping_df = pd.read_csv(name_mapping)
    mapping_df.dicom_id = mapping_df.dicom_id.astype(str)
    mapping_df.hecktor_id = mapping_df.hecktor_id.astype(str)
    mapping_dict = mapping_df.set_index("hecktor_id").to_dict()["dicom_id"]
    dicom_walker = DicomWalker(cores=12)
    studies = dicom_walker(input_folders)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    mapping_dicom_to_hecktor = {i: k for k, i in mapping_dict.items()}

    with open(list_cases_file) as f:
        list_cases_per_annotator = json.load(f)

    for annotator in list_cases_per_annotator.keys():
        folder = output_folder / annotator
        folder.mkdir(exist_ok=True)
        list_cases = [
            mapping_dict[c] for c in list_cases_per_annotator[annotator]
        ]
        studies_per_annotator = [
            s for s in studies if s.patient_id in list_cases
        ]
        found_cases = [s.patient_id for s in studies_per_annotator]
        missing_cases = [c for c in list_cases if c not in found_cases]
        if len(missing_cases) > 0:
            logger.warning(f"{annotator} missing cases: {missing_cases}")

        for s in studies_per_annotator:
            folder_patient = folder / mapping_dicom_to_hecktor[s.patient_id]
            folder_patient.mkdir(exist_ok=True)
            for volume in s.volume_files:
                folder_image = folder_patient / volume.modality
                folder_image.mkdir(exist_ok=True)
                for path in volume.dicom_paths:
                    copy(path, folder_image / Path(path).name)


if __name__ == '__main__':
    input_paths = [
        "/media/val/Windows/Users/valen/Documents/work/CHB",
        "/media/val/Windows/Users/valen/Documents/work/mda_test",
        "/home/val/python_wkspce/hecktor/data/hecktor2022/dicom/usz/USZ",
    ]

    list_cases = "/home/val/python_wkspce/hecktor/data/hecktor2022/cases_per_annotators.json"
    output_path = "/home/val/python_wkspce/hecktor/data/hecktor2022/dicom_for_annotators/"
    name_mapping = "/home/val/python_wkspce/hecktor/data/hecktor2022/name_mapping_test.csv"
    main(input_paths, output_path, list_cases, name_mapping)
