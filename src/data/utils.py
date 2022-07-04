from pathlib import Path
from datetime import datetime
import shutil
import warnings
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.ndimage.measurements import label as cc_label
import SimpleITK as sitk
from scipy.ndimage import affine_transform

from src.data.bounding_box import bbox_auto

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def get_datetime_from_dcm(s):
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
        shutil.move(f, archive_folder / f.name)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in patient_ids:
        move_gtv_one_patient(input_folder,
                             archive_folder,
                             patient_id,
                             label="GTVt")
        move_gtv_one_patient(input_folder,
                             archive_folder,
                             patient_id,
                             label="GTVn")


def find_missing_gtvt(patient_id, archive_folder, labels=None):
    files = [
        f for f in Path(archive_folder).rglob(f"{patient_id}__GTV*")
        if f.name.split("__")[1] in labels
    ]
    files.sort(key=get_datetime_from_filename)
    gtvt = files[-1]
    logger.info("Found GTVt: {}".format(gtvt))
    return


def combine_vois(
    input_folder,
    output_folder,
    archive_folder,
    center="montreal",
    voi_mapping=None,
):
    if voi_mapping:
        _combine_vois_with_mapping(input_folder, output_folder, archive_folder,
                                   voi_mapping)
        return
    if "mda" in center.lower():
        labels_gtvt = [
            "GTVp", "GTVP", "GTV_P", "GTVp_KW", "GTV_7", "GTVp_YK",
            "GTVp_YK_SA", "GTVp_KW_SA", "GTVp_SA", "GTV-P", "GTVp-YK", "GTVp2"
        ]
        labels_gtvn = [
            "GTVn1", "GTV_N1", "GTVn2", "GTV_N2", "GTV_N7", "GTV_N6", "GTV_N4",
            "GTV_N3", "GTV_N8", "GTV_N5", "GTVn3_SA", "GTVn1_SA", "GTVn2_SA",
            "GTVn3", "GTVn5", "GTVn4", "GTVn4_SA", "GTVn9", "GTVn8", "GTVn6",
            "GTVn7", "GTNn2", "GTVn01", "GTVn13", "GTVn14", "GTVn10", "GTVn12",
            "GTVn15", "GTVn11", "GTVn5_SA", "GTVn6_SA", "GTV_N10", "GTV_N11",
            "GTV_N9", "GTV_N_1", "GTV_N_2"
        ]

        _combine_vois_with_labels(
            input_folder,
            output_folder,
            archive_folder,
            labels_gtvt=labels_gtvt,
            labels_gtvn=labels_gtvn,
        )
    else:
        _combine_vois(input_folder, output_folder, archive_folder)


def _combine_vois(input_folder, output_folder, archive_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in tqdm(patient_ids, desc="Combining VOIs"):
        gtvts = [f for f in input_folder.rglob(f"{patient_id}__GTVt*")]
        gtvns = [f for f in input_folder.rglob(f"{patient_id}__GTVn*")]
        if len(gtvts) == 0:
            labels_fallback = ["GTV", "GTV_t", "GTVp"]
            logger.warning(
                f"No GTVt found for {patient_id}"
                f", trying to find GTVt among the name {labels_fallback}")
            try:
                gtvts = [
                    find_missing_gtvt(patient_id,
                                      archive_folder,
                                      labels=labels_fallback)
                ]
            except Exception as e:
                logger.error(f"No GTVt found for {patient_id}")
                continue
            continue
        mask = sitk.ReadImage(str(gtvts[0].resolve()))
        array = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvts:
            array += sitk.GetArrayFromImage(sitk.ReadImage(str(voi.resolve())))
        array_gtvn = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvns:
            array_gtvn += sitk.GetArrayFromImage(
                sitk.ReadImage(str(voi.resolve())))
        array[array != 0] = 1
        array[array_gtvn != 0] = 2
        output = sitk.GetImageFromArray(array)
        output.SetSpacing(mask.GetSpacing())
        output.SetDirection(mask.GetDirection())
        output.SetOrigin(mask.GetOrigin())
        sitk.WriteImage(output, str(output_folder / f"{patient_id}.nii.gz"))


def _combine_vois_with_mapping(input_folder, output_folder, archive_folder,
                               voi_mapping):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in tqdm(patient_ids, desc="Combining VOIs"):
        gtvs = [f for f in input_folder.rglob(f"{patient_id}__*")]
        labels_gtvt = voi_mapping[patient_id]["GTVt"]
        labels_gtvn = voi_mapping[patient_id]["GTVn"]
        gtvts = [f for f in gtvs if f.name.split("__")[1] in labels_gtvt]
        gtvns = [f for f in gtvs if f.name.split("__")[1] in labels_gtvn]
        if len(gtvts) == 0:
            logger.warning(f"No GTVt found for {patient_id}")
            mask = sitk.ReadImage(str(gtvns[0].resolve()))
            array = np.zeros_like(sitk.GetArrayFromImage(mask))
            array_gtvn = np.zeros_like(sitk.GetArrayFromImage(mask))
            for voi in gtvns:
                array_gtvn += sitk.GetArrayFromImage(
                    sitk.ReadImage(str(voi.resolve())))
            array[array != 0] = 1
            array[array_gtvn != 0] = 2
            output = sitk.GetImageFromArray(array)
            output.SetSpacing(mask.GetSpacing())
            output.SetDirection(mask.GetDirection())
            output.SetOrigin(mask.GetOrigin())
            sitk.WriteImage(output,
                            str(output_folder / f"{patient_id}.nii.gz"))
            continue

        mask = sitk.ReadImage(str(gtvts[0].resolve()))
        array = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvts:
            array += sitk.GetArrayFromImage(sitk.ReadImage(str(voi.resolve())))
        array_gtvn = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvns:
            array_gtvn += sitk.GetArrayFromImage(
                sitk.ReadImage(str(voi.resolve())))
        array[array != 0] = 1
        array[array_gtvn != 0] = 2
        output = sitk.GetImageFromArray(array)
        output.SetSpacing(mask.GetSpacing())
        output.SetDirection(mask.GetDirection())
        output.SetOrigin(mask.GetOrigin())
        sitk.WriteImage(output, str(output_folder / f"{patient_id}.nii.gz"))


def _combine_vois_with_labels(input_folder,
                              output_folder,
                              archive_folder,
                              labels_gtvt=None,
                              labels_gtvn=None):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in tqdm(patient_ids, desc="Combining VOIs"):
        gtvs = [f for f in input_folder.rglob(f"{patient_id}__*")]
        gtvts = [f for f in gtvs if f.name.split("__")[1] in labels_gtvt]
        gtvns = [f for f in gtvs if f.name.split("__")[1] in labels_gtvn]
        if len(gtvts) == 0:
            logger.warning(f"No GTVt found for {patient_id}")
            mask = sitk.ReadImage(str(gtvns[0].resolve()))
            array = np.zeros_like(sitk.GetArrayFromImage(mask))
            array_gtvn = np.zeros_like(sitk.GetArrayFromImage(mask))
            for voi in gtvns:
                array_gtvn += sitk.GetArrayFromImage(
                    sitk.ReadImage(str(voi.resolve())))
            array[array != 0] = 1
            array[array_gtvn != 0] = 2
            output = sitk.GetImageFromArray(array)
            output.SetSpacing(mask.GetSpacing())
            output.SetDirection(mask.GetDirection())
            output.SetOrigin(mask.GetOrigin())
            sitk.WriteImage(output,
                            str(output_folder / f"{patient_id}.nii.gz"))
            continue

        mask = sitk.ReadImage(str(gtvts[0].resolve()))
        array = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvts:
            array += sitk.GetArrayFromImage(sitk.ReadImage(str(voi.resolve())))
        array_gtvn = np.zeros_like(sitk.GetArrayFromImage(mask))
        for voi in gtvns:
            array_gtvn += sitk.GetArrayFromImage(
                sitk.ReadImage(str(voi.resolve())))
        array[array != 0] = 1
        array[array_gtvn != 0] = 2
        output = sitk.GetImageFromArray(array)
        output.SetSpacing(mask.GetSpacing())
        output.SetDirection(mask.GetDirection())
        output.SetOrigin(mask.GetOrigin())
        sitk.WriteImage(output, str(output_folder / f"{patient_id}.nii.gz"))


def sort_vois(
    input_folder,
    output_folder,
    archive_folder,
    center="montreal",
    voi_mapping=None,
):
    if voi_mapping:
        _sort_vois_with_mapping(input_folder, output_folder, archive_folder,
                                voi_mapping)
        return
    if "mda" in center.lower():
        _sort_vois_mda(input_folder, output_folder, archive_folder)
    elif "chup" in center.lower():
        _sort_vois_chup(input_folder, output_folder, archive_folder)
    else:
        _sort_vois(input_folder, output_folder, archive_folder)


def _sort_vois_with_mapping(input_folder, output_folder, archive_folder,
                            voi_mapping):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    archive_folder = Path(archive_folder)
    voi_files_to_move = [
        f for f in Path(input_folder).rglob("*RTSTRUCT*")
        if "PT" in f.name or ")." in f.name
    ]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in patient_ids:
        labels = set([
            f.name.split("__")[1]
            for f in input_folder.rglob(f"{patient_id}__*RTSTRUCT*")
        ])
        try:
            labels = [
                l for l in labels if l in voi_mapping[patient_id]["GTVt"]
                or l in voi_mapping[patient_id]["GTVn"]
            ]
        except KeyError:
            logger.warning(f"No mapping found for {patient_id}")
            continue
        for label in labels:
            move_gtv_one_patient(input_folder,
                                 output_folder,
                                 archive_folder,
                                 patient_id,
                                 label=label)

    voi_files_to_move = [f for f in Path(input_folder).rglob("*RTSTRUCT*")]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)


def _sort_vois_chup(input_folder, output_folder, archive_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    archive_folder = Path(archive_folder)
    voi_files_to_move = [
        f for f in Path(input_folder).rglob("*RTSTRUCT*")
        if "PT" in f.name or ")." in f.name
    ]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in patient_ids:
        labels = set([
            f.name.split("__")[1]
            for f in input_folder.rglob(f"{patient_id}__*RTSTRUCT*")
        ])
        for label in labels:
            move_gtv_one_patient(input_folder,
                                 output_folder,
                                 archive_folder,
                                 patient_id,
                                 label=label)

    voi_files_to_move = [f for f in Path(input_folder).rglob("*RTSTRUCT*")]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)


def _sort_vois(input_folder, output_folder, archive_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    archive_folder = Path(archive_folder)
    voi_files_to_move = [
        f for f in Path(input_folder).rglob("*RTSTRUCT*")
        if "PT" in f.name or ")." in f.name
    ]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)

    patient_ids = list(
        set([f.name.split("__")[0] for f in input_folder.rglob("*")]))

    for patient_id in patient_ids:
        labels = set([
            f.name.split("__")[1]
            for f in input_folder.rglob(f"{patient_id}__GTV*")
            if "GTVn" in f.name or "GTVt" in f.name
        ])
        for label in labels:
            move_gtv_one_patient(input_folder,
                                 output_folder,
                                 archive_folder,
                                 patient_id,
                                 label=label)

    voi_files_to_move = [f for f in Path(input_folder).rglob("*RTSTRUCT*")]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)


def _sort_vois_mda(input_folder, output_folder, archive_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    archive_folder = Path(archive_folder)
    voi_files_to_move = [
        f for f in Path(input_folder).rglob("*RTSTRUCT*")
        if "PT" in f.name or ")" in f.name
    ]
    for f in voi_files_to_move:
        shutil.move(f, archive_folder / f.name)


def move_gtv_one_patient(input_folder,
                         output_folder,
                         archive_folder,
                         patient_id,
                         label="GTVt"):
    vois = [f for f in input_folder.rglob(f"{patient_id}__{label}__*")]
    vois.sort(key=get_datetime_from_filename)
    for voi in vois[:-1]:
        shutil.move(voi, archive_folder / voi.name)

    shutil.move(vois[-1], output_folder / vois[-1].name)


def get_datetime_from_filename(file):
    return datetime.strptime(
        file.name.split("__")[-1].split(".")[0], "%Y-%m-%d_%H-%M-%S")


def move_gtv_one_patient_old(input_folder,
                             archive_folder,
                             patient_id,
                             label="GTVt"):
    voi_files = [
        f for f in Path(input_folder).rglob(f"*{label}*")
        if patient_id == f.name.split("__")[0]
    ]
    # if len(voi_files) == 0:
    # voi_files = [
    #     f for f in Path(input_folder).rglob("*RTSTRUCT*")
    #     if patient_id == f.name.split("__")[0]
    # ]
    # if len(voi_files) == 0:
    #     warnings.warn(f"patient {patient_id} has no VOI")
    #     continue
    if len(voi_files) == 0:
        warnings.warn(f"patient {patient_id} has no {label}")
        return

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
        shutil.move(f, archive_folder / f.name)


def correct_names(input_folder, output_folder, mapping, center="montreal"):
    if "montreal" in center.lower():
        _correct_names_montreal(input_folder, output_folder)
    else:
        _correct_names(input_folder, output_folder, mapping)


def map_names_hecktor(patient_id, mapping, center="montreal"):
    if "montreal" in center.lower():
        return _map_names_hecktor_montreal(patient_id)
    else:
        return _map_names_hecktor(patient_id, mapping)


def _map_names_hecktor(patient_id, mapping):
    mapping_df = pd.read_csv(mapping)
    mapping_df.dicom_id = mapping_df.dicom_id.astype(str)
    mapping_df.hecktor_id = mapping_df.hecktor_id.astype(str)
    mapping_dict = mapping_df.set_index("dicom_id").to_dict()["hecktor_id"]

    try:
        new_id = mapping_dict[patient_id]
    except KeyError:
        new_id = "No HECKTOR ID"
    return new_id


def _map_names_hecktor_montreal(patient_id):
    return patient_id[3:]


def _correct_names(input_folder, output_folder, mapping):
    input_folder = Path(input_folder)
    mapping_df = pd.read_csv(mapping)
    mapping_df.dicom_id = mapping_df.dicom_id.astype(str)
    mapping_df.hecktor_id = mapping_df.hecktor_id.astype(str)
    mapping_dict = mapping_df.set_index("dicom_id").to_dict()["hecktor_id"]

    files = [f for f in input_folder.rglob("*.nii.gz")]
    for file in files:
        patient_id = file.name.split("__")[0].split(".")[0]
        try:
            new_patient_id = mapping_dict[patient_id]
        except KeyError:
            warnings.warn(f"{patient_id} not found in mapping")
            continue
        if "CT" in file.name or "PT" in file.name:
            modality = file.name.split("__")[1]
            new_name = f"{new_patient_id}__{modality}.nii.gz"
        else:
            new_name = f"{new_patient_id}.nii.gz"

        shutil.copy(file, output_folder / new_name)


def _correct_names_montreal(input_folder, output_folder):
    input_folder = Path(input_folder)

    files = [f for f in input_folder.rglob("*.nii.gz")]
    for file in files:
        patient_id = file.name.split("__")[0].split(".")[0]
        new_patient_id = patient_id[3:]

        if "CT" in file.name or "PT" in file.name:
            modality = file.name.split("__")[1]
            new_name = f"{new_patient_id}__{modality}.nii.gz"
        else:
            new_name = f"{new_patient_id}.nii.gz"

        shutil.copy(file, output_folder / new_name)


def _correct_names_chuv(input_folder, mapping):
    input_folder = Path(input_folder)
    mapping_df = pd.read_csv(mapping)
    mapping_df["dicom_id"] = mapping_df["dicom_id"].map(
        lambda x: x.lstrip("P"))
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

        if patient_id.startswith("P"):
            patient_id = patient_id[1:]
        patient_id = patient_id.lstrip("0")
        # if "GTV" in modality:
        #     modality = "gtvt"
        try:
            new_name = (mapping_dict[patient_id] + "_" + modality.lower() +
                        ".nii.gz")
        except KeyError:
            warnings.warn(f"{patient_id} not found in mapping")
            continue
        file.rename(file.parent / new_name)


def remove_extra_components(
    mask,
    patient_id,
    threshold=50,
    label=1,
):
    array_output = sitk.GetArrayFromImage(mask)
    array = (array_output == label).astype(np.uint8)
    array_output[array_output == label] = 0
    if np.sum(array) == 0:
        return mask

    labels_dict = {1: "GTVp", 2: "GTVn"}
    voxel_volume = np.prod(mask.GetSpacing())
    array_label, num_features = cc_label(array, structure=np.ones((3, 3, 3)))
    volume_initial = int(np.sum(array)) * voxel_volume
    components = np.array(list(range(1, num_features + 1)))
    volumes = np.array(
        [np.sum(array_label == n) * voxel_volume for n in components])
    mono_slice = np.array(
        [np.max(np.sum(array_label == n, axis=0)) == 1 for n in components])

    if num_features > 1:
        logger.info(
            f"VOI {labels_dict[label]} for {patient_id} has {num_features} components"
            f" the smallest one has a volume of {np.min(volumes):0.2f} [mm3], "
            f"and {np.count_nonzero(mono_slice)} components are one only one slice"
        )
    components_to_keep = components[(volumes > threshold)
                                    & np.logical_not(mono_slice)]
    array = np.zeros_like(array)
    for c in components_to_keep:
        array += (array_label == c).astype(np.uint8)
    volume_final = int(np.sum(array)) * voxel_volume
    if volume_initial != volume_final:
        logger.warning(f"VOI {labels_dict[label]} for patient {patient_id} "
                       f"was corrected."
                       f" Volume initial: {volume_initial} [mm3]"
                       f" -> final total volume: {volume_final} [mm3]")
    array_output[array != 0] = label
    array_output = array_output.astype(np.uint8)
    output = sitk.GetImageFromArray(array_output)
    output.SetDirection(mask.GetDirection())
    output.SetOrigin(mask.GetOrigin())
    output.SetSpacing(mask.GetSpacing())
    return output


def correct_images_direction(image_folder, mask_folder):
    image_folder = Path(image_folder)
    files = [
        f for f in mask_folder.rglob("*.nii.gz") if "_corrected" in f.name
    ]
    for f in tqdm(files, desc="Correcting images direction"):
        mask = sitk.ReadImage(str(f))
        patient_id = f.name.split(".")[0].strip("_corrected")
        if mask.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1):
            continue
        mask = correct_direction(mask)
        logger.warning(f"Correcting direction of {f.name}")
        images_path = [p for p in image_folder.rglob(patient_id + "*")]
        for image_path in images_path:
            image = sitk.ReadImage(str(image_path))
            image = correct_direction(image)
            output_path = image_folder / (image_path.name.split(".")[0] +
                                          "_corrected.nii.gz")
            sitk.WriteImage(image, str(output_path))
        output_path = mask_folder / (patient_id + "_corrected.nii.gz")
        sitk.WriteImage(mask, str(output_path))


def correct_direction(image):
    if image.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1):
        return image
    array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    direction = np.reshape(np.array(image.GetDirection()), [3, 3])
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    coordinate_matrix = np.zeros((4, 4))
    coordinate_matrix[:3, :3] = direction * spacing
    coordinate_matrix[:3, 3] = origin
    coordinate_matrix[3, 3] = 1

    new_origin = np.zeros((3, ))
    for k in range(3):
        if direction[k, k] < 1:
            new_origin[k] = -array.shape[k] * spacing[k] + origin[k]
        else:
            new_origin[k] = origin[k]
    corrected_matrix = np.zeros((4, 4))
    corrected_matrix[:3, :3] = np.eye(3) * spacing
    corrected_matrix[:3, 3] = new_origin
    corrected_matrix[3, 3] = 1
    matrix = np.linalg.inv(coordinate_matrix) @ corrected_matrix
    # matrix = np.linalg.inv(corrected_matrix) @ coordinate_matrix
    array = affine_transform(
        array,
        matrix=matrix[:3, :3],
        output_shape=array.shape,
        offset=matrix[:3, 3],
        order=1,
        mode="nearest",
    )
    array = np.transpose(array, (2, 1, 0))
    output_image = sitk.GetImageFromArray(array)
    output_image.SetOrigin(image.GetOrigin())
    output_image.SetSpacing(image.GetSpacing())
    return output_image


def clean_vois(input_folder):
    input_folder = Path(input_folder)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((1, 1, 1))
    files = [
        f for f in input_folder.rglob("*.nii.gz") if not "_corrected" in f.name
    ]
    for f in tqdm(files, desc="Cleaning VOIs"):
        patient_id = f.name.split(".")[0]
        mask = sitk.ReadImage(str(f.resolve()))
        mask = remove_extra_components(mask, patient_id, label=1)
        mask = remove_extra_components(mask, patient_id, label=2)
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
