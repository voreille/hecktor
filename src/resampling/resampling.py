import os

import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk

from src.resampling.utils import (get_sitk_volume_from_np,
                                  get_np_volume_from_sitk)


class Resampler():
    def __init__(self,
                 bb_df,
                 output_folder,
                 order,
                 resampling=None,
                 logger=None):
        super().__init__()
        self.bb_df = bb_df
        self.output_folder = output_folder
        self.resampling = resampling
        self.order = order
        self.logger = logger

    def __call__(self, f, resampling=None):
        if resampling is None:
            resampling = self.resampling
        patient_name = f.split('/')[-1].split('_')[0]
        # patient_folder = os.path.join(self.output_folder, patient_name)
        # if not os.path.exists(patient_folder):
        #     os.mkdir(patient_folder)
        # output_file = os.path.join(patient_folder, f.split('/')[-1])
        output_file = os.path.join(self.output_folder, f.split('/')[-1])
        bb = (self.bb_df.loc[patient_name, 'x1'], self.bb_df.loc[patient_name,
                                                                 'y1'],
              self.bb_df.loc[patient_name, 'z1'], self.bb_df.loc[patient_name,
                                                                 'x2'],
              self.bb_df.loc[patient_name, 'y2'], self.bb_df.loc[patient_name,
                                                                 'z2'])
        print('Resampling patient {}'.format(patient_name))

        resample_and_crop(f,
                          output_file,
                          bb,
                          resampling=resampling,
                          order=self.order)


def resample_and_crop(input_file,
                      output_file,
                      bounding_box,
                      resampling=(1.0, 1.0, 1.0),
                      order=3):
    np_volume, pixel_spacing, origin = get_np_volume_from_sitk(
        sitk.ReadImage(input_file))
    resampling = np.asarray(resampling)
    # If one value of resampling is negative replace it with the original value
    for i in range(len(resampling)):
        if resampling[i] == -1:
            resampling[i] = pixel_spacing[i]
        elif resampling[i] < 0:
            raise ValueError(
                'Resampling value cannot be negative, except for -1')

    if 'gtv' in input_file or 'GTV' in input_file:
        np_volume = resample_np_binary_volume(np_volume, origin, pixel_spacing,
                                              resampling, bounding_box)
    else:
        np_volume = resample_np_volume(np_volume,
                                       origin,
                                       pixel_spacing,
                                       resampling,
                                       bounding_box,
                                       order=order)

    origin = np.asarray([bounding_box[0], bounding_box[1], bounding_box[2]])
    sitk_volume = get_sitk_volume_from_np(np_volume, resampling, origin)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_file)
    writer.SetImageIO("NiftiImageIO")
    writer.Execute(sitk_volume)


def resample_np_volume(np_volume,
                       origin,
                       current_pixel_spacing,
                       resampling_px_spacing,
                       bounding_box,
                       order=3):

    zooming_matrix = np.identity(3)
    zooming_matrix[0, 0] = resampling_px_spacing[0] / current_pixel_spacing[0]
    zooming_matrix[1, 1] = resampling_px_spacing[1] / current_pixel_spacing[1]
    zooming_matrix[2, 2] = resampling_px_spacing[2] / current_pixel_spacing[2]

    offset = ((bounding_box[0] - origin[0]) / current_pixel_spacing[0],
              (bounding_box[1] - origin[1]) / current_pixel_spacing[1],
              (bounding_box[2] - origin[2]) / current_pixel_spacing[2])

    output_shape = np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing

    np_volume = affine_transform(np_volume,
                                 zooming_matrix,
                                 offset=offset,
                                 mode='mirror',
                                 order=order,
                                 output_shape=output_shape.astype(int))

    return np_volume


def grid_from_spacing(start, spacing, n):
    return np.asarray([start + k * spacing for k in range(n)])


def resample_np_binary_volume(np_volume, origin, current_pixel_spacing,
                              resampling_px_spacing, bounding_box):

    x_old = grid_from_spacing(origin[0], current_pixel_spacing[0],
                              np_volume.shape[0])
    y_old = grid_from_spacing(origin[1], current_pixel_spacing[1],
                              np_volume.shape[1])
    z_old = grid_from_spacing(origin[2], current_pixel_spacing[2],
                              np_volume.shape[2])

    output_shape = (np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing).astype(int)

    x_new = grid_from_spacing(bounding_box[0], resampling_px_spacing[0],
                              output_shape[0])
    y_new = grid_from_spacing(bounding_box[1], resampling_px_spacing[1],
                              output_shape[1])
    z_new = grid_from_spacing(bounding_box[2], resampling_px_spacing[2],
                              output_shape[2])
    interpolator = RegularGridInterpolator((x_old, y_old, z_old),
                                           np_volume,
                                           method='nearest',
                                           bounds_error=False,
                                           fill_value=0)
    x, y, z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))

    return interpolator(pts).reshape(output_shape)
