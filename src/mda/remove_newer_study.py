from genericpath import exists
from datetime import datetime
import gc
from pathlib import Path
import logging
import pickle
import shutil
from multiprocessing import Pool

import click
from tqdm import tqdm
import pydicom as pdcm
from okapy.dicomconverter.dicom_walker import DicomWalker

path_in = "/home/val/python_wkspce/hecktor/data/mda_train_sorted/PETCT"
path_out = "/home/val/python_wkspce/hecktor/data/mda_train_sorted/additional"

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
def main(input_folder, output_folder):
    logger.info('Starting to walk through all the DICOM files')
    for patient_folder in Path(input_folder).iterdir():
        study_dir = [k for k in patient_folder.iterdir()]
        if len(study_dir) == 1:
            continue
        dates_list = [study_date(study) for study in study_dir]
        yo = list(zip(study_dir, dates_list))
        yo.sort(key=lambda x: x[1])
        for y in yo[1:]:
            shutil.move(y[0], path_out)


def study_date(folder):
    files = [f for f in folder.rglob("PT*")]
    data = pdcm.dcmread(files[0], stop_before_pixels=True)
    return datetime.strptime(
        data[0x00080022].value + data[0x00080032].value.split('.')[0],
        "%Y%m%d%H%M%S")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
