# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.evaluation.scores import dice
from src.resampling.utils import get_np_volume_from_sitk
from src.resampling.resampling import resample_np_binary_volume

# %%
prediction_folder = '/home/val/python_wkspce/hecktor/data/resampled'
groundtruth_folder = '/home/val/python_wkspce/hecktor/data/hecktor_nii'
bb_filepath = '/home/val/python_wkspce/hecktor/data/bbox.csv'

# %%
# List of the files in the validation
prediction_files = [f for f in Path(prediction_folder).rglob('*gtvt.nii.gz')]

# The list is sorted, so it will match the list of ground truth files
prediction_files.sort(key=lambda x: x.name.split('_')[0])

# List of the patient_id in the validation
patient_name_predictions = [f.name.split('_')[0] for f in prediction_files]

# %%
# List of the ground truth files
groundtruth_files = [
    f for f in Path(groundtruth_folder).rglob('*gtvt.nii.gz')
    if f.name.split('_')[0] in patient_name_predictions
]
# The list is sorted to match the validation list
groundtruth_files.sort(key=lambda x: x.name.split('_')[0])

# %%
# The bounding boxes will be used to compute the Dice score within.
bb_df = pd.read_csv(bb_filepath).set_index('PatientID')

# %%
# DataFrame to store the results
results_df = pd.DataFrame()

for i, f in enumerate(reversed(prediction_files)):
    patient_name = f.name.split('_')[0]
    gt_file = [k for k in groundtruth_files if k.name[:7] == patient_name][0]
    print('Evaluating patient {}'.format(patient_name))
    bb = (bb_df.loc[patient_name,
                    'x1'], bb_df.loc[patient_name,
                                     'y1'], bb_df.loc[patient_name, 'z1'],
          bb_df.loc[patient_name,
                    'x2'], bb_df.loc[patient_name,
                                     'y2'], bb_df.loc[patient_name, 'z2'])

    sitk_pred = sitk.ReadImage(str(f.resolve()))
    sitk_gt = sitk.ReadImage(str(gt_file.resolve()))
    # Transform from SimpleITK to numpy, otherwise the bounding boxes axis are swapped
    np_pred, px_spacing_pred, origin_pred = get_np_volume_from_sitk(sitk_pred)
    np_gt, px_spacing_gt, origin_gt = get_np_volume_from_sitk(sitk_gt)

    # Resample back to the original resolution and crop in the bounding box
    np_pred = resample_np_binary_volume(np_pred, origin_pred, px_spacing_pred,
                                        px_spacing_gt, bb)
    np_gt = resample_np_binary_volume(np_gt, origin_gt, px_spacing_gt,
                                      px_spacing_gt, bb)

    # Store the results
    results_df = results_df.append(
        {
            'PatientID': patient_name,
            'Dice Score': dice(np_gt, np_pred),
            'Original Resolution': px_spacing_gt,
        },
        ignore_index=True)

# %%
results_df['Dice Score'].values


# %%
