import time

import pandas as pd

from survival_metrics import concordance_index


class HecktorSurvEvaluator:
    def __init__(self, ground_truth_file, round_number=1):
        """Evaluator for the Hecktor Challenge

        Args:
            ground_truth_folder (str): the path to the file 
                                       containing the ground truth survival time.
            round_number (int, optional): the round number. Defaults to 1.
        """
        self.gt_df = pd.read_csv(ground_truth_file).set_index("PatientID")
        self.round = round_number

    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
            - submission_file_path : local file path of the submitted file
            - aicrowd_submission_id : A unique id representing the submission
            - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]
        is_concordant = client_payload["is_concordant"]
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

        if is_concordant:
            concordance_factor = 1
        else:
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


if __name__ == "__main__":
    ground_truth_file = ""
    _client_payload = {}
    _client_payload["submission_file_path"] = ""
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234
    _client_payload["is_concordant"] = True

    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    aicrowd_evaluator = HecktorSurvEvaluator(ground_truth_file)

    # Evaluate
    start = time.process_time()
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print("Time to compute the sample {}".format(time.process_time() - start))
    print(result)
    print(result["message"])
