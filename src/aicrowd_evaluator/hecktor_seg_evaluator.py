import os
import zipfile
from pathlib import Path
import warnings
from shutil import rmtree
import time

import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from segmentation_metrics import compute_segmentation_scores


class HecktorSegEvaluator:
    def __init__(self,
                 ground_truth_folder,
                 bounding_boxes_filepath,
                 extraction_folder="data/extraction/",
                 round_number=1):
        """Evaluator for the Hecktor Challenge

        Args:
            ground_truth_folder (str): the path to the folder 
                                       containing the ground truth segmentation.
            bounding_boxes_filepath (str): the path to the csv file which defines
                                           the bounding boxes for each patient.
            extraction_folder (str, optional): the path to the folder where the 
                                               extraction of the .zip submission 
                                               will take place. Defaults to "data/tmp/".
                                               This folder has to be created beforehand.
            round_number (int, optional): the round number. Defaults to 1.
        """
        self.groud_truth_folder = Path(ground_truth_folder)
        self.round = round_number
        self.extraction_folder = Path(extraction_folder)
        self.bounding_boxes_filepath = Path(bounding_boxes_filepath)

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
            - submission_file_path : local file path of the submitted file
            - aicrowd_submission_id : A unique id representing the submission
            - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
        submission_extraction_folder = self.extraction_folder / (
            'submission' + str(aicrowd_submission_id) + '/')

        submission_extraction_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(Path(submission_file_path).resolve()),
                             "r") as zip_ref:
            zip_ref.extractall(str(submission_extraction_folder.resolve()))

        groundtruth_paths = [
            f for f in self.groud_truth_folder.rglob("*.nii.gz")
        ]
        bb_df = pd.read_csv(str(
            self.bounding_boxes_filepath.resolve())).set_index("PatientID")

        results_df = pd.DataFrame()

        missing_patients = list()
        unresampled_patients = list()
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        for path in tqdm(groundtruth_paths):
            patient_id = path.name[:7]
            prediction_files = [
                f
                for f in self.extraction_folder.rglob(patient_id + "*.nii.gz")
            ]
            if len(prediction_files) > 1:
                raise Exception(
                    "There is too many prediction files for patient {}".format(
                        patient_id))
            elif len(prediction_files) == 0:
                results_df = results_df.append(
                    {
                        "dice_score": 0,
                        "hausdorff_distance_95": np.inf,
                        "recall": 0,
                        "precision": 0,
                    },
                    ignore_index=True)

                missing_patients.append(patient_id)
                continue

            bb = np.array([
                bb_df.loc[patient_id, "x1"], bb_df.loc[patient_id, "y1"],
                bb_df.loc[patient_id, "z1"], bb_df.loc[patient_id, "x2"],
                bb_df.loc[patient_id, "y2"], bb_df.loc[patient_id, "z2"]
            ])

            image_gt = sitk.ReadImage(str(path.resolve()))
            image_pred = sitk.ReadImage(str(prediction_files[0].resolve()))

            resampler.SetReferenceImage(image_gt)
            resampler.SetOutputOrigin(bb[:3])

            voxel_spacing = np.array(image_gt.GetSpacing())
            output_size = np.round(
                (bb[3:] - bb[:3]) / voxel_spacing).astype(int)
            resampler.SetSize([int(k) for k in output_size])

            # Crop to the bonding box and/or resample to the original spacing
            spacing = image_gt.GetSpacing()
            if spacing != image_pred.GetSpacing():
                unresampled_patients.append(patient_id)

            image_gt = resampler.Execute(image_gt)
            image_pred = resampler.Execute(image_pred)

            results_df = results_df.append(
                compute_segmentation_scores(
                    sitk.GetArrayFromImage(image_gt),
                    sitk.GetArrayFromImage(image_pred),
                    spacing,
                ),
                ignore_index=True,
            )

        _result_object = {
            "dice_score": results_df["dice_score"].mean(),
            "hausdorff_distance_95":
            results_df["hausdorff_distance_95"].median(),
            "recall": results_df["recall"].mean(),
            "precision": results_df["precision"].mean(),
        }

        rmtree(str(submission_extraction_folder.resolve()))
        messages = list()
        if len(unresampled_patients) > 0:
            messages.append(
                f"The following patient(s) was/were not resampled back"
                f" to the original resolution: {unresampled_patients}."
                f"\nWe applied a nearest neighbor resampling.\n")

        if len(missing_patients) > 0:
            messages.append(
                f"The following patient(s) was/were missing: {missing_patients}."
                f"\nA score of 0 was attributed to them")
        _result_object["message"] = "".join(messages)
        return _result_object


if __name__ == "__main__":
    ground_truth_folder = ""
    bounding_boxes_file = ""
    _client_payload = {}
    _client_payload["submission_file_path"] = ""
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = HecktorSegEvaluator(ground_truth_folder,
                                            bounding_boxes_file)
    # Evaluate
    start = time.process_time()
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print("Time to compute the sample {}".format(time.process_time() - start))
    print(result)
    print(result["message"])
