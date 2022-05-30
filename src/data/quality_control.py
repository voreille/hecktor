from pathlib import Path
from multiprocessing import Pool

import click
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]

default_input_folder = project_dir / "data/hecktor2022/processed"
default_output_path = project_dir / "data/hecktor2022/qc.csv"


@click.command()
@click.argument('input_folder',
                type=click.Path(),
                default=default_input_folder)
@click.argument('output_path', type=click.Path(), default=default_output_path)
@click.option("--cores", type=click.INT, default=None)
def main(input_folder, cores):
    input_folder = Path(input_folder)
    vois = [f for f in input_folder.rglob("*/labels/*")]
    patient_ids = set([f.name.split(".")[0] for f in vois])
    processor = PatientProcessor(input_folder)
    if cores:
        with Pool(cores) as p:
            result = list(
                tqdm(p.imap(processor, patient_ids), total=len(patient_ids)))
    else:
        result = []
        for patient_id in patient_ids:
            result.append(processor(patient_id))

    df = pd.concat(result)

    print(patient_ids)


class PatientProcessor():

    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetInterpolator(sitk.sitkLinear)

    def __call__(self, patient_id):
        ct_paths = [f for f in self.input_folder.rglob(f"*{patient_id}__CT*")]
        pt_paths = [f for f in self.input_folder.rglob(f"*{patient_id}__PT*")]
        mask_paths = [
            f for f in self.input_folder.rglob(f"*labels/{patient_id}*")
        ]
        ct = sitk.ReadImage(ct_paths[0])
        pt = sitk.ReadImage(pt_paths[0])
        mask = sitk.ReadImage(mask_paths[0])

        self.resampler.SetOutputSpacing(ct.GetSpacing())
        self.resampler.SetOutputDirection(ct.GetDirection())
        pt = self.resampler.Execute(pt)
        array_ct = sitk.GetArrayFromImage(ct)
        array_pt = sitk.GetArrayFromImage(pt)
        array_mask = sitk.GetArrayFromImage(mask)



if __name__ == '__main__':
    main()
