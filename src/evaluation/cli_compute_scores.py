from pathlib import Path

import click
import logging
import pandas as pd
import SimpleITK as sitk

from src.evaluation.scores import dice


@click.command()
@click.argument('ground_truth_folder',
                type=click.Path(exists=True),
                default='data/gt')
@click.argument('prediction_folder',
                type=click.Path(exists=True),
                default='data/results/')
@click.argument('output_filepath',
                type=click.Path(),
                default='data/results/results.csv')
def main(prediction_folder, ground_truth_folder, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Computing Dice scores')
    results_df = pd.DataFrame(columns=['PatientID', 'Dice score'])
    groundtruth_files_list = [
        f for f in Path(ground_truth_folder).rglob('*GTVt.nii')
    ]
    for path in groundtruth_files_list:
        np_image_gt = sitk.GetArrayFromImage(
            sitk.ReadImage(str(path.resolve())))
        patient_name = path.name[0:10]
        prediction_files = [
            f
            for f in Path(prediction_folder).rglob(patient_name + '*GTVt.nii')
        ]
        if len(prediction_files) > 1:
            raise RuntimeError(
                'There is too many prediction files for patient {}'.format(
                    patient_name))
        # check_segmentation(sitk_pred)
        np_image_pred = sitk.GetArrayFromImage(
            sitk.ReadImage(str(prediction_files[0].resolve())))
        score = dice(np_image_gt, np_image_pred)
        results_df = results_df.append(
            {
                'PatientID': patient_name,
                'Dice score': score
            },
            ignore_index=True)

    results_df.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
