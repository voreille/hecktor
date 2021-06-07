from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk

# Default paths
# path_in = 'data/hecktor_nii/'
path_in = '/home/valentin/python_wkspce/hecktor/data/hecktor2021_train/hecktor_nii'
path_out = 'data/resampled/'
# path_bb = 'data/bbox.csv'
path_bb = '/home/valentin/python_wkspce/hecktor/data/hecktor2021_train/hecktor2021_bbox_training.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
@click.argument('bounding_boxes_file', type=click.Path(), default=path_bb)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1),
              help='Expect 3 positive floats describing the output '
              'resolution of the resampling. To avoid resampling '
              'on one or more dimension a value of -1 can be fed '
              'e.g. --resampling 1.0 1.0 -1 will resample the x '
              'and y axis at 1 mm/px and left the z axis untouched.')
@click.option('--order',
              type=click.INT,
              nargs=1,
              default=3,
              help='The order of the spline interpolation used to resample')
def main(input_folder, output_folder, bounding_boxes_file, cores, resampling,
         order):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    print('resampling is {}'.format(str(resampling)))
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    patient_list = [f.name.split("_")[0] for f in input_folder.rglob("*_ct*")]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)

    def resample_one_patient(p):
        bb = np.array([
            bb_df.loc[p, 'x1'], bb_df.loc[p, 'y1'], bb_df.loc[p, 'z1'],
            bb_df.loc[p, 'x2'], bb_df.loc[p, 'y2'], bb_df.loc[p, 'z2']
        ])
        size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        ct = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "_ct*")][0].resolve()))
        pt = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "_pt*")][0].resolve()))
        gtvt = sitk.ReadImage(
            str([f for f in input_folder.rglob(p + "_gtvt*")][0].resolve()))
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        gtvt = resampler.Execute(gtvt)
        sitk.WriteImage(ct, str(
            (output_folder / (p + "_ct.nii.gz")).resolve()))
        sitk.WriteImage(pt, str(
            (output_folder / (p + "_pt.nii.gz")).resolve()))
        sitk.WriteImage(gtvt,
                        str((output_folder / (p + "_gtvt.nii.gz")).resolve()))

    # with Pool(cores) as p:
    #     p.map(resample_one_patient, patient_list)

    for patient in patient_list:
        resample_one_patient(patient)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
