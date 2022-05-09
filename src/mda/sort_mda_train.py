from genericpath import exists
import gc
from pathlib import Path
import logging
import pickle
import shutil
from multiprocessing import Pool

import click
import pandas as pd
from tqdm import tqdm
import pydicom as pdcm
from okapy.dicomconverter.dicom_walker import DicomWalker

path_in = "/media/val/Windows/Users/valen/Documents/work/mda_train"
path_out = "/home/val/python_wkspce/hecktor/data/mda_train_debug/PETCT"

logger = logging.getLogger(__name__)

WRITE_STUDY = False


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
def main(input_folder, output_folder):
    logger.info('Starting to walk through all the DICOM files')

    clinical_info = pd.read_excel(
        "/home/val/python_wkspce/hecktor/data/Patient and Treatment Characteristics (2).xls"
    )
    # patient_ids = clinical_info[clinical_info["Site"] ==
    #                             "Oropharynx"]["TCIA code"].values
    patient_ids = ["HNSCC-01-0022"]

    dicom_walker = DicomWalker(
        cores=8,
        additional_dicom_tags=["ImageType"],
    )
    if WRITE_STUDY:
        studies = dicom_walker(input_folder)
        with open("studies.pkl", "wb") as f:
            pickle.dump(studies, f)
    else:
        with open("studies.pkl", "rb") as f:
            studies = pickle.load(f)

    studies = [s for s in studies if s.patient_id in patient_ids]
    for s in tqdm(studies):
        process_study(s, output_folder)

    # def f(study):
    #     return process_study(study, output_folder)

    # with Pool(8) as p:
    #     successes = p.map(f, studies)

    print("yoi")


def copy_volume(volume, destination):
    for f in volume.dicom_paths:
        shutil.copy(f, destination)


def process_study(study, output_dir):
    pt_volumes = [
        f for f in study.volume_files
        if f.modality == "PT" # and "DERIVED" not in f.dicom_header.image_type
    ]

    ct_volumes = [
        f for f in study.volume_files
        if f.modality == "CT" # and "DERIVED" not in f.dicom_header.image_type
    ]

    masks = [f for f in study.mask_files]
    # if len(ct_volumes) == 0 or len(masks) == 0:
    #     return 0

    if len(ct_volumes) == 0:
        return 0

    # pt_volumes = list()
    # for f in pt_volumes_initial:
    #     try:
    #         f.read_without_pixel()
    #     except Exception as e:
    #         patient_id = f.dicom_header.patient_id
    #         logger.warning(f"Exception {e} \nfor patient {patient_id}")
    #         continue
    #     # f.read()
    #     pt_volumes.append(f)

    # pt_volumes = [
    #     f for f in pt_volumes if f.slices[0].Units in ["CNTS", "GML", "BQML"]
    # ]
    if len(pt_volumes) == 0:
        return 0

    # n_slices_max = max([f.expected_n_slices for f in pt_volumes])
    # pt_volumes = [f for f in pt_volumes if f.expected_n_slices == n_slices_max]
    patient_id = pt_volumes[0].dicom_header.patient_id
    study_id = pt_volumes[0].dicom_header.study_instance_uid.replace(".", "_")

    output_dir_study = Path(output_dir) / f"{patient_id}/{study_id}"
    output_dir_study.mkdir(exist_ok=True, parents=True)

    for v in pt_volumes:
        copy_volume(v, output_dir_study)
        del v.slices
        v.slices = None

    for v in ct_volumes:
        copy_volume(v, output_dir_study)
        del v.slices
        v.slices = None

    for v in masks:
        copy_volume(v, output_dir_study)
        del v.slices
        v.slices = None

    gc.collect()
    return 1


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
