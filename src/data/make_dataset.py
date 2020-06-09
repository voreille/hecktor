import os
from os.path import join
import warnings
import logging

import click
import SimpleITK as sitk
import pydicom as pdcm
import pandas as pd
import yaml

from src.data.bounding_box import bbox_auto
from src.data.dicom_conversion import (correct_patient_name,
                                       convert_dicom_to_nifty,
                                       MissingWeightException)

with open('settings.yaml') as file:
    settings = yaml.full_load(file)

path_dicom = settings['path']['dicom']
path_nii = settings['path']['nii']
path_bb = settings['path']['bb_file']
label_rtstruct = settings['voi']


@click.command()
@click.argument('input_filepath',
                type=click.Path(exists=True),
                default=path_dicom)
@click.argument('output_filepath', type=click.Path(), default=path_nii)
@click.argument('bb_filepath', type=click.Path(), default=path_bb)
@click.option('--extension', type=click.STRING, default='.nii.gz')
@click.option('--subfolders', is_flag=True)
def main(input_filepath, output_filepath, bb_filepath, extension, subfolders):
    """Command Line Interface to make the conversion from DICOM to NIFTI,
       computing Hounsfield units and SUV for CT and PT respectively.

    Args:
        input_filepath (str): Path to the folder containing the DICOM files
        output_filepath (str): Path to the folder where the NIFTI will be
                               stored.
        bb_filepath (str): Path where to write the .csv containing the
                           bounding boxes.
        extension (str): The extension used for the NIFTI file (.nii or
                         .nii.gz)
        subfolders (bool): Wether to create one subfolder for each patient
                           or to store everything within the same directory.

    Raises:
        ValueError: Raised if more than 1 RTSTRUCT file is found
    """
    logger = logging.getLogger(__name__)
    logger.info('Converting Dicom to Nifty')
    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetImageIO('NiftiImageIO')
    bb_df = pd.DataFrame(
        columns=['PatientID', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'])
    for (dirpath, dirnames, filenames) in os.walk(input_filepath):
        files_list_ct = [join(dirpath, k) for k in filenames if '.CT.' in k]
        files_list_pt = [join(dirpath, k) for k in filenames if '.PT.' in k]
        files_list_rtstruct = [
            join(dirpath, k) for k in filenames if '.RTSTRUCT.' in k
        ]
        if len(files_list_rtstruct) > 1:
            raise ValueError('There is more than 1 RTSTRUCT file')
        if len(files_list_ct) > 0:
            patient_name = dirpath.split('/')[-1]
            patient_name = correct_patient_name(patient_name)
            if subfolders:
                path_output_folder = join(
                    correct_patient_name(output_filepath), patient_name)
            else:
                path_output_folder = correct_patient_name(output_filepath)
            if not os.path.exists(path_output_folder):
                os.mkdir(path_output_folder)

            logger.info('Creating folder {}'.format(output_filepath))
            logger.info('Converting the CT ')

            _ = convert_dicom_to_nifty(files_list_ct,
                                       path_output_folder,
                                       modality='CT',
                                       sitk_writer=sitk_writer,
                                       rtstruct_file=files_list_rtstruct[0],
                                       labels_rtstruct=label_rtstruct,
                                       extension=extension)
            logger.info('Converting the PT of {}'.format(patient_name))
            try:
                (np_pt, px_spacing_pt,
                 im_pos_patient_pt) = convert_dicom_to_nifty(
                     files_list_pt,
                     path_output_folder,
                     modality='PT',
                     sitk_writer=sitk_writer,
                     rtstruct_file=None,
                     extension=extension)
            except MissingWeightException:
                header_ct = pdcm.dcmread(files_list_ct[0],
                                         stop_before_pixels=True)
                patient_weight = header_ct.PatientWeight
                if patient_weight is None:
                    patient_weight = 75.0  # From Martin's code
                    warnings.warn(
                        "Cannot find the weight of the patient, hence it "
                        "is approximated to be 75.0 kg")

                (np_pt, px_spacing_pt,
                 im_pos_patient_pt) = convert_dicom_to_nifty(
                     files_list_pt,
                     path_output_folder,
                     modality='PT',
                     sitk_writer=sitk_writer,
                     rtstruct_file=None,
                     patient_weight_from_ct=patient_weight,
                     extension=extension)
            bb = bbox_auto(np_pt, px_spacing_pt, im_pos_patient_pt)
            bb_df = bb_df.append(
                {
                    'PatientID': patient_name,
                    'x1': bb[0],
                    'x2': bb[1],
                    'y1': bb[2],
                    'y2': bb[3],
                    'z1': bb[4],
                    'z2': bb[5],
                },
                ignore_index=True)
    bb_df.to_csv(bb_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
