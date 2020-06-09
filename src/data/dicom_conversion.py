import warnings
from os.path import join
from datetime import time, datetime

import numpy as np
from skimage.draw import polygon
import SimpleITK as sitk
import pydicom as pdcm
from pydicom.tag import Tag


def convert_dicom_to_nifty(input_filepaths,
                           output_folder,
                           modality='CT',
                           sitk_writer=None,
                           rtstruct_file=None,
                           labels_rtstruct=['GTVt'],
                           patient_weight_from_ct=None,
                           extension='.nii',
                           dtype_image=np.float32,
                           dtype_mask=np.uint8):
    """Function to convert the dicom files contained in input_filepaths to one
       NIFTI image.

    Args:
        input_filepaths (list): list of the dicom paths
        output_folder (str): path to the output folder where to store the
                             NIFTI file.
        modality (str, optional): The modality of the DICOM, it is used to
                                  obtain the correct physical values
                                  (Hounsfield unit for the CT and SUV for the
                                  PT). Defaults to 'CT'.
        sitk_writer (sitk.WriteImage(), optional): The SimpleITK object used
                                                   to write an array to the
                                                   NIFTI format. Defaults to
                                                   None.
        rtstruct_file (str, optional): Path to the RTSTRUCT file associated
                                       with the current image. Defaults to
                                       None.
        labels_rtstruct (list, optional): List of label to extract from the
                                          RTSTRUCT. Defaults to ['GTVt'].
        patient_weight_from_ct (float, optional): If the patient's weight is
                                                  missing from the PT DICOM
                                                  it can be provided through
                                                  this argument. Defaults to
                                                  None.
        extension (str, optional): The extension in which to save the NIFTI.
                                   Defaults to '.nii'.
        dtype_image (numpy.dtype, optional): The dtype in which to save the
                                             image. Defaults to np.float32.
        dtype_mask (numpy.dtype, optional): The dtype in which to save the
                                            segmentation. Defaults to np.uint8.

    Raises:
        MissingWeightException: Error to alert when the weight is missing from
                                the PT, to compute the SUV.
        RuntimeError: Error to alert when one or more slices are missing
        ValueError: Raised when a modality or a unit (for the PT) is not
                    handled.

    Returns:
        numpy.array: The numpy image, used to compute the bounding boxes
    """
    slices = [pdcm.read_file(dcm) for dcm in input_filepaths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if modality == 'PT':
        if slices[0].PatientWeight is None:
            if hasattr(slices[0], 'PatientsWeight'):
                patient_weight = float(slices[0].PatientsWeight)
            elif patient_weight_from_ct is not None:
                patient_weight = patient_weight_from_ct
            else:
                raise MissingWeightException(
                    'Cannot compute SUV the weight is missing')
        else:
            patient_weight = float(slices[0].PatientWeight)

    # Check if all the slices come from the same serie
    same_serie_uid = True
    serie_uid = slices[0].SeriesInstanceUID
    for s in slices:
        same_serie_uid *= serie_uid == s.SeriesInstanceUID

    if not same_serie_uid:
        raise RuntimeError('A slice comes from another serie')

    axial_positions = np.asarray([k.ImagePositionPatient[2] for k in slices])
    # Compute redundant slice positions
    ind2rm = [
        ind for ind in range(len(axial_positions))
        if axial_positions[ind] == axial_positions[ind - 1]
    ]
    # Check if there is redundancy in slice positions and remove them
    if len(ind2rm) > 0:
        slices = [k for i, k in enumerate(slices) if i not in ind2rm]
        axial_positions = np.asarray(
            [k.ImagePositionPatient[2] for k in slices])

    slice_spacing = (slices[1].ImagePositionPatient[2] -
                     slices[0].ImagePositionPatient[2])

    pixel_spacing = np.asarray([
        slices[0].PixelSpacing[0],
        slices[0].PixelSpacing[1],
        slice_spacing,
    ])

    if modality == 'CT':
        np_image = get_physical_values_ct(slices, dtype=dtype_image)
    elif modality == 'PT':
        np_image = get_physical_values_pt(slices,
                                          patient_weight,
                                          dtype=dtype_image)
    else:
        raise ValueError('The modality {} is not supported'.format(modality))

    position_final_slice = (
        len(slices) - 1) * slice_spacing + slices[0].ImagePositionPatient[2]
    # Test whether some slices are missing
    if not is_approx_equal(position_final_slice,
                           float(slices[-1].ImagePositionPatient[2])):
        if (position_final_slice - axial_positions[-1]) / slice_spacing < 1.5:
            # If only one slice is missing
            diff = np.asarray([
                not is_approx_equal(
                    float(axial_positions[ind]) -
                    float(axial_positions[ind - 1]) - slice_spacing, 0)
                for ind in range(1, len(axial_positions))
            ])
            ind2interp = int(np.where(diff)[0])
            new_slice = (np_image[:, :, ind2interp] +
                         np_image[:, :, ind2interp + 1]) * 0.5
            new_slice = new_slice[..., np.newaxis]
            np_image = np.concatenate(
                (np_image[..., :ind2interp], new_slice, np_image[...,
                                                                 ind2interp:]),
                axis=2)
            warnings.warn(
                "One slice is missing, we replaced it by linear interpolation")
        else:
            # if more than one slice are missing
            raise RuntimeError('Multiple slices are missing')

    image_position_patient = [float(k) for k in slices[0].ImagePositionPatient]
    sitk_image = get_sitk_volume_from_np(np_image, pixel_spacing,
                                         image_position_patient)

    output_filepath = join(
        output_folder,
        correct_patient_name(str(slices[0].PatientName)) + '_' +
        modality.lower() + extension)
    sitk_writer.SetFileName(output_filepath)
    sitk_writer.Execute(sitk_image)
    if rtstruct_file is not None:
        masks = get_masks(rtstruct_file,
                          labels=labels_rtstruct,
                          image_position_patient=image_position_patient,
                          axial_positions=axial_positions,
                          pixel_spacing=pixel_spacing,
                          shape=np_image.shape,
                          dtype=dtype_image)
        for label, np_mask in masks:
            output_filepath_mask = output_filepath.split(
                '.')[0] + '_' + label.lower() + extension
            sitk_mask = get_sitk_volume_from_np(np_mask, pixel_spacing,
                                                image_position_patient)
            sitk_writer.SetFileName(output_filepath_mask)
            sitk_writer.Execute(sitk_mask)
    return np.transpose(np_image,
                        (1, 0, 2)), pixel_spacing, image_position_patient


def get_sitk_volume_from_np(np_image, pixel_spacing, image_position_patient):
    trans = (2, 0, 1)
    sitk_image = sitk.GetImageFromArray(np.transpose(np_image, trans))
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(image_position_patient)
    return sitk_image


def is_approx_equal(x, y, tolerance=0.05):
    return abs(x - y) <= tolerance


class MissingWeightException(RuntimeError):
    pass


def correct_patient_name(patient_name):
    output = patient_name.replace("HN-CHUS-", "CHUS")
    output = output.replace("HN-CHUM-", "CHUM")
    output = output.replace("HN-HGJ-", "CHGJ")
    output = output.replace("HN-HMR-", "CHMR")
    output = output.replace("HN-CHUV-", "CHUV")
    return output


def get_masks(rtstruct_file,
              labels=['GTVt', 'GTVn'],
              image_position_patient=None,
              axial_positions=None,
              pixel_spacing=None,
              shape=None,
              dtype=np.int8):
    contours = read_structure(rtstruct_file, labels=labels)
    return get_mask_from_contour(contours,
                                 image_position_patient,
                                 axial_positions,
                                 pixel_spacing,
                                 shape,
                                 dtype=dtype)


def read_structure(rtstruct_file, labels=['GTVt', 'GTVn']):
    structure = pdcm.read_file(rtstruct_file)
    contours = []
    for i, roi_seq in enumerate(structure.StructureSetROISequence):
        contour = {}
        for label in labels:
            if roi_seq.ROIName == label:
                contour['color'] = structure.ROIContourSequence[
                    i].ROIDisplayColor
                contour['number'] = structure.ROIContourSequence[
                    i].ReferencedROINumber
                contour['name'] = roi_seq.ROIName
                assert contour['number'] == roi_seq.ROINumber
                contour['contours'] = [
                    s.ContourData
                    for s in structure.ROIContourSequence[i].ContourSequence
                ]
                contours.append(contour)

    return contours


def get_mask_from_contour(contours,
                          image_position_patient,
                          axial_positions,
                          pixel_spacing,
                          shape,
                          dtype=np.uint8):
    z = np.asarray(axial_positions)
    pos_r = image_position_patient[1]
    spacing_r = pixel_spacing[1]
    pos_c = image_position_patient[0]
    spacing_c = pixel_spacing[0]

    output = []
    for con in contours:
        mask = np.zeros(shape, dtype=dtype)
        for current in con['contours']:
            nodes = np.array(current).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = np.where((nodes[0, 2] - 0.001 < z)
                               & (z < nodes[0, 2] + 0.001))[0][0]
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            if len(rr) > 0 and len(cc) > 0:
                if np.max(rr) > 512 or np.max(cc) > 512:
                    raise Exception("The RTSTRUCT file is compromised")

            mask[rr, cc, z_index] = 1
        output.append((con['name'], mask))
    return output


def get_physical_values_ct(slices, dtype=np.float32):
    image = list()
    for s in slices:
        image.append(
            float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept))
    return np.stack(image, axis=-1).astype(dtype)


def get_physical_values_pt_old(slices):
    units = slices[0].Units
    s = slices[0]
    if units == 'BQML':
        datetime_acquisition = datetime.strptime(
            s[Tag(0x00080022)].value + s[Tag(0x00080032)].value.split('.')[0],
            "%Y%m%d%H%M%S")
        datetime_serie = datetime.strptime(
            s[Tag(0x00080021)].value + s[Tag(0x00080031)].value.split('.')[0],
            "%Y%m%d%H%M%S")
        if datetime_serie < datetime_acquisition and datetime_serie > datetime(
                1950, 1, 1):
            pass
        else:
            pass
        return get_suv_from_bqml(slices)
    elif units == 'CNTS':
        return get_suv_philips(slices)
    else:
        raise ValueError('The {} units is not handled'.format(units))


def get_physical_values_pt(slices, patient_weight, dtype=np.float32):
    s = slices[0]
    units = s.Units
    if units == 'BQML':

        acquisition_datetime = datetime.strptime(
            s[Tag(0x00080022)].value + s[Tag(0x00080032)].value.split('.')[0],
            "%Y%m%d%H%M%S")
        serie_datetime = datetime.strptime(
            s[Tag(0x00080021)].value + s[Tag(0x00080031)].value.split('.')[0],
            "%Y%m%d%H%M%S")

        try:
            if (serie_datetime < acquisition_datetime) and (
                    serie_datetime > datetime(1950, 1, 1)):
                scan_datetime = serie_datetime
            else:
                scan_datetime_value = s[Tag(0x0009100d)].value
                if isinstance(scan_datetime_value, bytes):
                    scan_datetime_str = scan_datetime_value.decode(
                        "utf-8").split('.')[0]
                elif isinstance(scan_datetime_value, str):
                    scan_datetime_str = scan_datetime_value.split('.')[0]
                else:
                    raise ValueError(
                        "The value of scandatetime is not handled")
                scan_datetime = datetime.strptime(scan_datetime_str,
                                                  "%Y%m%d%H%M%S")

            start_time_str = s.RadiopharmaceuticalInformationSequence[
                0].RadiopharmaceuticalStartTime
            start_time = time(int(start_time_str[0:2]),
                              int(start_time_str[2:4]),
                              int(start_time_str[4:6]))
            start_datetime = datetime.combine(scan_datetime.date(), start_time)
            decay_time = (scan_datetime - start_datetime).total_seconds()
        except KeyError:
            warnings.warn("Estimation of time decay for SUV"
                          " computation from average parameters")
            decay_time = 1.75 * 3600  # From Martin's code
        return get_suv_from_bqml(slices,
                                 decay_time,
                                 patient_weight,
                                 dtype=dtype)

    elif units == 'CNTS':
        return get_suv_philips(slices, dtype=dtype)
    else:
        raise ValueError('The {} units is not handled'.format(units))


def get_suv_philips(slices, dtype=np.float32):
    image = list()
    suv_scale_factor_tag = Tag(0x70531000)
    for s in slices:
        im = (float(s.RescaleSlope) * s.pixel_array +
              float(s.RescaleIntercept)) * float(s[suv_scale_factor_tag].value)
        image.append(im)
    return np.stack(image, axis=-1).astype(dtype)


def get_suv_from_bqml(slices, decay_time, patient_weight, dtype=np.float32):
    # Get SUV from raw PET
    image = list()
    for s in slices:
        pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
        half_life = float(
            s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        total_dose = float(
            s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        decay = 2**(-decay_time / half_life)
        actual_activity = total_dose * decay

        im = pet * patient_weight * 1000 / actual_activity
        image.append(im)
    return np.stack(image, axis=-1).astype(dtype)
