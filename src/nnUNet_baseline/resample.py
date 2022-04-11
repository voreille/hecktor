from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk

# Default paths
path_in = 'data/nnUNet_raw_data_base/nnUNet_raw_data/Task500_hecktor2021'
path_bb_test = 'data/hecktor2021_test/hecktor2021_bbox_testing.csv'
path_bb_train = 'data/hecktor2021_train/hecktor2021_bbox_training.csv'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('bounding_boxes_file_train',
                type=click.Path(),
                default=path_bb_train)
@click.argument('bounding_boxes_file_test',
                type=click.Path(),
                default=path_bb_test)
@click.option('--cores',
              type=click.INT,
              default=12,
              help='The number of workers for parallelization.')
def main(input_folder, bounding_boxes_file_train, bounding_boxes_file_test,
         cores):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation of degree 3.
        The resampling spacing is inferred from the CT image.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    logger = logging.getLogger(__name__)
    logger.info('Resampling')

    input_folder = Path(input_folder).resolve()

    bb_df = pd.concat(
        [
            pd.read_csv(bounding_boxes_file_test).set_index("PatientID"),
            pd.read_csv(bounding_boxes_file_train).set_index("PatientID"),
        ],
        axis=0,
    )

    patient_list = [
        f.name.split("_")[0] for f in input_folder.rglob("*_0000.nii.gz")
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    def resample_one_patient(p):
        bb = np.array([
            bb_df.loc[p, 'x1'], bb_df.loc[p, 'y1'], bb_df.loc[p, 'z1'],
            bb_df.loc[p, 'x2'], bb_df.loc[p, 'y2'], bb_df.loc[p, 'z2']
        ])
        path_ct = [f for f in input_folder.rglob(p + "_0000.nii.gz")][0]
        path_pt = [f for f in input_folder.rglob(p + "_0001.nii.gz")][0]
        path_gtvt = [
            f for f in input_folder.rglob("*labelsTr/" + p + ".nii.gz")
        ]
        ct = sitk.ReadImage(str(path_ct))
        pt = sitk.ReadImage(str(path_pt))

        if len(path_gtvt) > 0:
            gtvt = sitk.ReadImage(str(path_gtvt[0]))
        else:
            gtvt = None

        resampling_spacing = ct.GetSpacing()
        size = np.round((bb[3:] - bb[:3]) / resampling_spacing).astype(int)
        resampler.SetOutputSpacing(resampling_spacing)
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        sitk.WriteImage(ct, str(path_ct))
        sitk.WriteImage(pt, str(path_pt))

        if gtvt is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            gtvt = resampler.Execute(gtvt)
            sitk.WriteImage(gtvt, str(path_gtvt[0]))

    for p in patient_list:
        resample_one_patient(p)
    # with Pool(cores) as p:
    #    p.map(resample_one_patient, patient_list)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
