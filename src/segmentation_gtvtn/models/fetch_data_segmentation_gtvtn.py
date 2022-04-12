from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np
import SimpleITK as sitk
import random


def get_tf_dataset_gtvtn(path_data_nii,
                         bbox_path,
                         modalities=['ct', 'pt'],
                         augment_shift=None,
                         isAugmentMirror=False,
                         augment_angles=None,
                         num_parallel_calls=None,
                         regex=None,
                         regex_in=True,
                         output_shape=(144, 144, 144),
                         isReturnPatientName=False,
                         task=None,
                         isTrain=True):
    """

    Args:
        path_data_nii (str): Path to the folder containing the original nii 
        bbox_path (str): the path to the bounding box
        augment_shift (float, optional): magnitute of random shifts. Defaults to None.
        isAugmentMirror (bool, optional): Whether to apply random sagittal mirroring. Defaults to False.
        augment_angles (tuple, optional): whether to apply random rotation, of maximal amplitude of
                                         (angle_x, angle_y, angle_z) where the angle are defined in degree.
        num_parallel_calls (int, optional): num of CPU for reading the data. If None tensorfow decides.
        regex (str, optional): Regex expression to filter patient names. Defaults to None.
                                If None no filtering is applied.
        regex_in (bool, optional): Wether to exclude or include the regex. Defaults to True.
        output_shape (tuple, optional): Output shape of the spatial domain. Defaults to (144,144,144).
        isReturnPatientName (bool, optional): Wether to return the patient name


    Returns:
        [type]: [description]
    """
    patient_folders = [
        str(f.resolve()) for f in Path(path_data_nii).glob("./*/")
    ]
    # load bboxes
    dfbbox = pd.read_csv(bbox_path)
    # Generate a tf.dataset for the paths to the folder
    patient_folders_ds = tf.data.Dataset.from_tensor_slices(patient_folders)
    # get the tf.dataset to return the path and the name of the patient
    patient_folders_ds = patient_folders_ds.map(tf_get_patient_name)

    # Filter according to regex and regex_in
    if regex:
        if regex_in:
            def f_regex(patient_name): return tf.strings.regex_full_match(
                patient_name, regex)
        else:
            def f_regex(patient_name): return tf.math.logical_not(
                tf.strings.regex_full_match(patient_name, regex))

        patient_folders_ds = patient_folders_ds.filter(
            lambda x, patient_name: f_regex(patient_name))

    def f(x, patient_name): return (*tf_parse_image(
        x,
        dfbbox,
        modalities=modalities,
        output_shape=output_shape,
        augment_shift=augment_shift,
        isAugmentMirror=isAugmentMirror,
        augment_angles=augment_angles,
        task=task,
        isTrain=isTrain
    ), patient_name)

    # Mapping the parsing function
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

    image_ds = patient_folders_ds.map(f, num_parallel_calls=num_parallel_calls)
    if isReturnPatientName is False:
        image_ds = image_ds.map(lambda x, y, z: (x, y))
    return image_ds


def tf_get_patient_name(patient_folder):
    return patient_folder, tf.strings.split(patient_folder, '/')[-1]


def tf_parse_image(patient_folder,
                   dfbbox,
                   modalities=['ct', 'pt'],
                   output_shape=(144, 144, 144),
                   augment_shift=None,
                   augment_angles=None,
                   isAugmentMirror=False,
                   task=None,
                   isTrain=True):
    def f(x): return parse_image(x,
                                 dfbbox,
                                 modalities=modalities,
                                 output_shape=output_shape,
                                 augment_shift=augment_shift,
                                 augment_angles=augment_angles,
                                 isAugmentMirror=isAugmentMirror)
    image, mask, mask_gtvn = tf.py_function(f,
                                            inp=[patient_folder],
                                            Tout=(tf.float32, tf.float32, tf.float32))

    if modalities == 'pt' or modalities == 'ct':
        image.set_shape(output_shape + (1, ))
    else:
        image.set_shape(output_shape + (2, ))
    mask.set_shape(output_shape + (1, ))
    mask_gtvn.set_shape(output_shape + (1, ))
    if task == 'GTVtn':
        return image, tf.concat([mask, mask_gtvn], axis=-1)
    elif task in ['GTVt', 'GTVn'] or isTrain == False:
        return image, (mask, mask_gtvn)
    else:
        return image, mask


def normalize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


def parse_image(patient_folder,
                dfbbox,
                modalities=['ct', 'pt'],
                output_shape=(144, 144, 144),
                clipping_range_ct=(-300, 300),
                augment_shift=None,
                isAugmentMirror=False,
                augment_angles=None,
                interp_order=1):
    """Parse the raw data of HECKTOR 2020
    """

    patient_folder_path = Path(patient_folder.numpy().decode("utf-8"))
    patient_name = patient_folder_path.name
    ct_sitk = sitk.ReadImage(
        str((patient_folder_path / (patient_name + "_ct.nii.gz")).resolve()), sitk.sitkFloat32)
    pt_sitk = sitk.ReadImage(
        str((patient_folder_path / (patient_name + "_pt.nii.gz")).resolve()), sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(
        str((patient_folder_path /
             (patient_name + "_gtvt.nii.gz")).resolve()), sitk.sitkFloat32)
    try:
        mask_sitk_gtvn = sitk.ReadImage(
            str((patient_folder_path /
                (patient_name + "_gtvn.nii.gz")).resolve()))
    except:
        mask_sitk_gtvn = None
    origin = dfbbox.loc[dfbbox['PatientID'] ==
                        patient_name, ['x1', 'y1', 'z1']].values
    resampler = sitk.ResampleImageFilter()
    if interp_order == 3:
        resampler.SetInterpolator(sitk.sitkBSpline)
    if augment_shift:
        origin += np.random.random_sample(
            3) * 2 * augment_shift - augment_shift

    resampler.SetOutputOrigin(tuple(*origin))
    resampler.SetOutputSpacing((1, 1, 1))
    resampler.SetSize(output_shape)

    if augment_angles:
        augment_angles = np.array(augment_angles) * np.pi / 180
        center = (origin + dfbbox.loc[dfbbox['PatientID'] == patient_name,
                                      ['x2', 'y2', 'z2']].values) / 2
        transform = sitk.Euler3DTransform()
        transform.SetCenter(np.squeeze(center))
        transform.SetRotation(*(
            np.random.random_sample(3) * 2 * augment_angles - augment_angles))
        resampler.SetTransform(transform)
    ct_sitk = resampler.Execute(ct_sitk)
    pt_sitk = resampler.Execute(pt_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_sitk = resampler.Execute(mask_sitk)
    if mask_sitk_gtvn is not None:
        mask_sitk_gtvn = resampler.Execute(mask_sitk_gtvn)

    def to_np(x): return np.transpose(sitk.GetArrayFromImage(x), (2, 1, 0))

    ct = to_np(ct_sitk)
    pt = to_np(pt_sitk)
    ct[ct < clipping_range_ct[0]] = clipping_range_ct[0]
    ct[ct > clipping_range_ct[1]] = clipping_range_ct[1]
    ct = (2 * ct - clipping_range_ct[1] -
          clipping_range_ct[0]) / (clipping_range_ct[1] - clipping_range_ct[0])

    pt = normalize_image(pt)
    if 'ct' in modalities and 'pt' in modalities:
        image = np.stack([ct, pt], axis=-1)
    elif modalities == 'ct':
        image = tf.expand_dims(ct, axis=-1)
    elif modalities == 'pt':
        image = tf.expand_dims(pt, axis=-1)
    mask = to_np(mask_sitk)[..., np.newaxis]
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    if mask_sitk_gtvn is not None:
        mask_gtvn = to_np(mask_sitk_gtvn)[..., np.newaxis]
        mask_gtvn[mask_gtvn >= 0.5] = 1
        mask_gtvn[mask_gtvn < 0.5] = 0
    else:
        mask_gtvn = np.zeros(mask.shape)
    if isAugmentMirror:
        if bool(random.getrandbits(1)):
            mask = np.flip(mask, axis=0)
            mask_gtvn = np.flip(mask_gtvn, axis=0)
            image = np.flip(image, axis=0)
    return image, mask, mask_gtvn


def get_ds(ds):
    images = list()
    masks = list()
    for elem in ds:
        if len(elem) < 3:
            names = None
        else:
            names = list()
        if elem[1][0].shape[-1] != 2 and len(elem[1]) < 2:
            masks_gtvn = None
        else:
            masks_gtvn = list()
        break

    for elem in ds:
        for image in elem[0].numpy():
            images.append(image)
        if len(elem[1]) > 1:
            for mask in elem[1][0]:
                masks.append(mask[..., 0:1].numpy())
            for mask_n in elem[1][1]:
                masks_gtvn.append(mask_n.numpy())
        else:
            for mask in elem[1]:
                masks.append(mask[..., 0:1].numpy())
                if mask.shape[-1] == 2:
                    masks_gtvn.append(mask[..., 1:2].numpy())
        if len(elem) > 2:
            for name in elem[2].numpy():
                names.append(name.decode("utf-8"))
    if names is not None:
        return np.asarray(images), np.asarray(masks), np.asarray(masks_gtvn), np.asarray(names)
    else:
        return np.asarray(images), np.asarray(masks), np.asarray(masks_gtvn)
