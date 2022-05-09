from genericpath import exists
from datetime import datetime
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

# path_in = "/home/val/python_wkspce/hecktor/data/mda_train_sorted/PETCT"
path_in = "/home/val/python_wkspce/hecktor/data/mda_train_sorted/PETCTRTSTRUCT"
path_out = "/home/val/python_wkspce/hecktor/data/mda_train_sorted/additional_patients2"

logger = logging.getLogger(__name__)


def main():
    logger.info('Starting to walk through all the DICOM files')

    clinical_info = pd.read_excel(
        "/home/val/python_wkspce/hecktor/data/Patient and Treatment Characteristics (2).xls"
    )
    patient_id = clinical_info[clinical_info["Site"] ==
                               "Oropharynx"]["TCIA code"].values
    for patient_dir in Path(path_in).iterdir():
        if patient_dir.name in patient_id:
            continue
        else:
            shutil.move(patient_dir, path_out)

    print("yo")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)
    main()
