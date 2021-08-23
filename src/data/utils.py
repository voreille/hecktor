from pathlib import Path
from datetime import datetime
from shutil import move
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label
import SimpleITK as sitk

from src.data.bounding_box import bbox_auto

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def get_datetime(s):
    return datetime.strptime(s.SeriesDate + s.SeriesTime.split('.')[0],
                             "%Y%m%d%H%M%S")


def move_extra_vois(input_folder, archive_folder):
    input_folder = Path(input_folder)
    archive_folder = Path(archive_folder)
    voi_files = [
        f for f in Path(input_folder).rglob("*RTSTRUCT*")
        if "PT" in f.name or ")." in f.name
    ]
    for f in voi_files:
        move(f, archive_folder / f.name)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in patient_ids:
        voi_files = [
            f for f in Path(input_folder).rglob("*GTV*")
            if patient_id == f.name.split("__")[0]
        ]
        if len(voi_files) == 0:
            voi_files = [
                f for f in Path(input_folder).rglob("*RTSTRUCT*")
                if patient_id == f.name.split("__")[0]
            ]
            if len(voi_files) == 0:
                warnings.warn(f"patient {patient_id} has no VOI")
                continue

        voi_datetimes = [
            datetime.strptime(
                f.name.split("__")[-1].split(".")[0], "%Y-%m-%d_%H-%M-%S")
            for f in voi_files
        ]
        voi_files_dates = list(zip(voi_files, voi_datetimes))
        voi_files_dates.sort(key=lambda x: x[1])
        voi_to_keep = voi_files_dates[-1][0]
        voi_to_move = [
            f for f in Path(input_folder).rglob("*RTSTRUCT*")
            if patient_id == f.name.split("__")[0] and f != voi_to_keep
        ]
        for f in voi_to_move:
            move(f, archive_folder / f.name)


def correct_names(input_folder, mapping):
    input_folder = Path(input_folder)
    mapping_df = pd.read_csv(mapping)
    mapping_dict = {
        k: i
        for k, i in zip(list(mapping_df["dicom_id"]),
                        list(mapping_df["hecktor_id"]))
    }
    files = [
        f for f in input_folder.rglob("*.nii.gz")
        if not f.name.startswith("CHU")
    ]
    for file in files:
        patient_id, modality = file.name.split("__")[:2]
        patient_id = patient_id.replace("_ORL", "")
        if "GTV" in modality:
            modality = "gtvt"
        new_name = (mapping_dict[patient_id] + "_" + modality.lower() +
                    ".nii.gz")
        file.rename(file.parent / new_name)


def remove_extra_components(mask, patient_id, threshold=0.01):
    array = sitk.GetArrayFromImage(mask)
    array_label, num_features = label(array)
    if num_features > 1:
        total_n_vox = int(np.sum(array))
        components = np.array(list(range(1, num_features + 1)))
        volumes = np.array([np.sum(array_label == n) for n in components])
        components_to_keep = components[volumes > threshold * total_n_vox]
        array = np.zeros_like(array)
        for c in components_to_keep:
            array += (array_label == c).astype(np.uint8)
        final_n_vox = int(np.sum(array))
        logger.warning(f"GTVt for patient {patient_id} "
                       f"has multiple components, keeping"
                       f"only the largest, total_voxels: {total_n_vox}"
                       f" -> final_voxels: {final_n_vox}")
        output = sitk.GetImageFromArray(array)
        output.SetDirection(mask.GetDirection())
        output.SetOrigin(mask.GetOrigin())
        output.SetSpacing(mask.GetSpacing())
        return output
    else:
        return mask


def clean_vois(input_folder):
    input_folder = Path(input_folder)
    for f in input_folder.rglob("*gtvt.nii.gz"):
        patient_id = f.name.split("_")[0]
        mask = sitk.ReadImage(str(f.resolve()))
        mask = remove_extra_components(mask, patient_id)
        filepath = str((f.parent / (f.name.split(".")[0] + "_corrected" +
                                    "".join(f.suffixes))).resolve())
        sitk.WriteImage(mask, filepath)


def compute_bbs(input_folder):
    input_folder = Path(input_folder)
    bb_df = pd.DataFrame()
    for file in input_folder.rglob("*pt.nii.gz"):
        patient_name = file.name.split("_")[0]
        pet_image = sitk.ReadImage(str(file.resolve()))
        bb = bbox_auto(pet_image)
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
    return bb_df.set_index("PatientID")
