import glob

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import SimpleITK as sitk


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default='data/processed')
@click.argument('output_file',
                type=click.Path(),
                default='data/original_resolution_ct.csv')
@click.option('--extension', type=click.STRING, default='.nii.gz')
def main(input_folder, output_file, extension):
    """Command Line Inteface used to generate a csv file containing the
       original voxel spacing.

    Args:
        input_folder (str): Path containing the NIFTI images in the original
                            resolution.
        output_file (str): Path where to store the csv.
        extension (str): String containing the extension of the NIFTI files ('.nii' or '.nii.gz')
    """

    resolution_dict = pd.DataFrame()
    for f in glob.glob(input_folder + '/**/*_ct' + extension, recursive=True):
        patient_name = f.split('/')[-2]
        sitk_image = sitk.ReadImage(f)
        px_spacing = sitk_image.GetSpacing()
        resolution_dict = resolution_dict.append(
            {
                'PatientID': patient_name,
                'Resolution_x': px_spacing[0],
                'Resolution_y': px_spacing[1],
                'Resolution_z': px_spacing[2]
            },
            ignore_index=True)

    resolution_dict.to_csv(output_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
