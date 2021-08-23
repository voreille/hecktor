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
from survival_metrics import concordance_index


class AIcrowdEvaluator:
    def __init__(
        self,
        ground_truth_segmentation_folder="data/ground_truth/segmentation/",
        ground_truth_survival_file="data/ground_truth/survival/hecktor2021_patient_endpoint_testing.csv",
        bounding_boxes_file="data/hecktor2021_bbox_testing.csv",
        extraction_folder="data/extraction/",
        round_number=1,
    ):
        """Evaluator for the Hecktor Challenge

        Args:
            ground_truth_folder (str): the path to the folder 
                                       containing the ground truth segmentation.
            ground_truth_survival_file (str): the path to the file 
                                       containing the ground truth survival time.
            bounding_boxes_file (str): the path to the csv file which defines
                                           the bounding boxes for each patient.
            extraction_folder (str, optional): the path to the folder where the 
                                               extraction of the .zip submission 
                                               will take place. Defaults to "data/tmp/".
                                               This folder has to be created beforehand.
            round_number (int, optional): the round number. Defaults to 1.
        """
        self.groud_truth_folder = Path(ground_truth_segmentation_folder)
        self.round = round_number
        self.extraction_folder = Path(extraction_folder)
        self.bounding_boxes_file = Path(bounding_boxes_file)
        self.gt_df = pd.read_csv(ground_truth_survival_file).set_index(
            "PatientID")

    def _evaluate_segmentation(self, client_payload, _context={}):
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
            self.bounding_boxes_file.resolve())).set_index("PatientID")

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
                f"\nA score of 0 and infinity were attributed to them "
                f"for the dice score and Hausdorff distance respectively.")
        _result_object["message"] = "".join(messages)
        return _result_object

    def _evaluate_survival(self, client_payload, _context={}):
        submission_file_path = client_payload["submission_file_path"]
        predictions_df = pd.read_csv(submission_file_path).set_index(
            "PatientID")

        if "Prediction" not in predictions_df.columns:
            raise RuntimeError("The 'Prediction' column is missing.")

        extra_patients = [
            p for p in predictions_df.index if p not in self.gt_df.index
        ]

        # Discard extra patient
        if len(extra_patients) > 0:
            predictions_df = predictions_df.drop(labels=extra_patients, axis=0)

        # Check for redundant entries
        if len(predictions_df.index) > len(list(set(predictions_df.index))):
            raise RuntimeError("One or more patients appear twice in the csv")

        # The following function concatenate the submission csv and the
        # ground truth and fill missing entries with NaNs. The missing
        # entries are then counted as non-concordant by the concordance_index
        # function
        df = pd.concat((self.gt_df, predictions_df), axis=1)
        missing_patients = list(df.loc[pd.isna(df['Prediction']), :].index)

        # Compute the c-index for anti-concordant prediction (e.g. risk score)
        concordance_factor = -1

        _result_object = {
            "concordance_index":
            concordance_index(
                df["Progression free survival"].values,
                concordance_factor * df["Prediction"],
                event_observed=df["Progression"],
            ),
        }

        messages = list()
        if len(missing_patients) > 0:
            messages = (f"The following patient(s) was/were missing"
                        f" : {missing_patients}\nThey were considered as "
                        f"non-concordant")
        if len(extra_patients) > 0:
            messages.append(
                f"The following patient(s) was/were dropped "
                f"(since they are not present in the test): {missing_patients}."
            )
        _result_object["message"] = "".join(messages)

        return _result_object

    def _get_evaluation_function(self, task_id, client_payload, _context={}):
        if task_id == "1":
            return self._evaluate_segmentation(client_payload,
                                               _context=_context)
        elif task_id == "2":
            return self._evaluate_survival(client_payload, _context=_context)
        else:
            raise ValueError(f"{task_id} is not recognized.")

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
        - submission_file_path : local file path of the submitted file
        - aicrowd_submission_id : A unique id representing the submission
        - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        task_id = os.environ["TASK_ID"]
        return self._get_evaluation_function(task_id,
                                             client_payload,
                                             _context=_context)


if __name__ == "__main__":
    ground_truth_segmentation_folder = ""
    ground_truth_survival_file = ""
    bounding_boxes_file = ""
    _client_payload = {}
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = AIcrowdEvaluator(
        ground_truth_segmentation_folder=ground_truth_segmentation_folder,
        ground_truth_survival_file=ground_truth_survival_file,
        bounding_boxes_file=bounding_boxes_file,
    )
    # Evaluate Survival
    _client_payload[
        "submission_file_path"] = ""
    os.environ["TASK_ID"] = "2"
    start = time.process_time()
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(f"Time to compute the sample for the survival"
          f" task: {time.process_time() - start} [s]")
    print(f"The c-index is {result['concordance_index']}")
    if result["message"] is not "":
        print(f"The message is:\n {result['message']}")

    # Evaluate Segmentation
    os.environ["TASK_ID"] = "1"
    _client_payload[
        "submission_file_path"] = ""
    start = time.process_time()
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(f"Time to compute the sample for the segmentation"
          f" task: {time.process_time() - start} [s]")
    print(f"The results are:\n"
          f" - average dice score {result['dice_score']}\n"
          f" - median hausdorff distance {result['hausdorff_distance_95']}\n"
          f" - average recall {result['recall']}\n"
          f" - average precision {result['precision']}")
    if result["message"] is not "":
        print(f"The message is:\n {result['message']}")
