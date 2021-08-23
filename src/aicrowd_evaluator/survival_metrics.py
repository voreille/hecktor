import numpy as np

def concordance_index(event_times, predicted_scores, event_observed=None):
    """
    Code adapted from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/concordance.py
    to account for missing values in the context of the HECKTOR Challenge
    Missing values are encoded by NaNs and are computed as non-concordant.
    """
    event_times, predicted_scores, event_observed = _preprocess_scoring_data(
        event_times, predicted_scores, event_observed)
    # num_correct, num_tied, num_pairs = _concordance_summary_statistics(
    #     event_times, predicted_scores, event_observed)
    num_correct, num_tied, num_pairs = _naive_concordance_summary_statistics(
        event_times, predicted_scores, event_observed)

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    """
    Code adapted from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/concordance.py
    to account for missing values in the context of the HECKTOR Challenge
    """
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _naive_concordance_summary_statistics(event_times, predicted_event_times,
                                          event_observed):
    """
    Code adapted from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/concordance.py
    to account for missing values in the context of the HECKTOR Challenge
    """
    def _valid_comparison(time_a, time_b, event_a, event_b):
        """True if times can be compared."""
        if time_a == time_b:
            # Ties are only informative if exactly one event happened
            return event_a != event_b
        if event_a and event_b:
            return True
        if event_a and time_a < time_b:
            return True
        if event_b and time_b < time_a:
            return True
        return False

    def _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b):
        if np.isnan(pred_a) or np.isnan(pred_b):
            # Missing values, same as random
            return (0, 1)
        if pred_a == pred_b:
            # Same as random
            return (0, 1)
        if pred_a < pred_b:
            return (time_a < time_b) or (time_a == time_b and event_a
                                         and not event_b), 0
        # pred_a > pred_b
        return (time_a > time_b) or (time_a == time_b and not event_a
                                     and event_b), 0

    num_pairs = 0.0
    num_correct = 0.0
    num_tied = 0.0

    for a, time_a in enumerate(event_times):
        pred_a = predicted_event_times[a]
        event_a = event_observed[a]
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if _valid_comparison(time_a, time_b, event_a, event_b):
                num_pairs += 1.0
                crct, ties = _concordance_value(time_a, time_b, pred_a, pred_b,
                                                event_a, event_b)
                num_correct += crct
                num_tied += ties

    return (num_correct, num_tied, num_pairs)


def _preprocess_scoring_data(event_times, predicted_scores, event_observed):
    """
    Code adapted from https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/concordance.py
    to account for missing values in the context of the HECKTOR Challenge
    """
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1
                                  or event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if predicted_scores.ndim == 2 and (predicted_scores.shape[0] == 1
                                       or predicted_scores.shape[1] == 1):
        # Flatten array
        predicted_scores = predicted_scores.ravel()

    if event_times.shape != predicted_scores.shape:
        raise ValueError(
            "Event times and predictions must have the same shape")
    if event_times.ndim != 1:
        raise ValueError("Event times can only be 1-dimensional: (n,)")

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError(
                "Observed events must be 1-dimensional of same length as event times"
            )
    # Commented out since we rely on NaNs to count missing patients
    # check for NaNs
    # for a in [event_times, predicted_scores, event_observed]:
    #     if np.isnan(a).any():
    #         raise ValueError(
    #             "NaNs detected in inputs, please correct or drop.")

    return event_times, predicted_scores, event_observed
