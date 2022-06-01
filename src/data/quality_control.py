from pathlib import Path
from multiprocessing import Pool
from itertools import product

import click
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics.featureextractor import RadiomicsFeatureExtractor

project_dir = Path(__file__).resolve().parents[2]
data_dir = Path(
    "/run/media/val/083C23E228226C35/work/hecktor2022/processed/mda_test")

default_input_folder = data_dir
default_output_path = project_dir / "data/hecktor2022/qc.csv"


@click.command()
@click.argument('input_folder',
                type=click.Path(),
                default=default_input_folder)
@click.argument('output_path', type=click.Path(), default=default_output_path)
@click.option("--cores", type=click.INT, default=None)
def main(input_folder, output_path, cores):
    input_folder = Path(input_folder)
    vois = [f for f in input_folder.rglob("*labels_renamed/*")]
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
    df.to_csv(output_path)


class PatientProcessor():

    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetInterpolator(sitk.sitkLinear)
        self.featureextractor = RadiomicsFeatureExtractor(
            str(project_dir / "src/data/qc_parameters.yaml"))

    def _clean_pyradiomics_outptut(self, output):
        return {k: v for k, v in output.items() if "iagnos" not in k}

    def _append_results(
        self,
        patient_id,
        features,
        results,
        modality="CT",
        voi="GTVp",
    ):
        features = self._clean_pyradiomics_outptut(features)
        results = results.append(
            {
                "patient_id": patient_id,
                "VOI": voi,
                "modality": modality,
                **features
            },
            ignore_index=True)
        return results

    def __call__(self, patient_id):
        ct_paths = [
            f for f in self.input_folder.rglob(
                f"*images_renamed/{patient_id}__CT*")
        ]
        pt_paths = [
            f for f in self.input_folder.rglob(
                f"*images_renamed/{patient_id}__PT*")
        ]
        mask_paths = [
            f
            for f in self.input_folder.rglob(f"*labels_renamed/{patient_id}*")
        ]
        ct = sitk.ReadImage(str(ct_paths[0]))
        pt = sitk.ReadImage(str(pt_paths[0]))
        mask = sitk.ReadImage(str(mask_paths[0]))

        self.resampler.SetOutputSpacing(ct.GetSpacing())
        self.resampler.SetOutputDirection(ct.GetDirection())
        pt = self.resampler.Execute(pt)

        results = pd.DataFrame()
        images = {"PT": pt, "CT": ct}
        vois_label = {"GTVp": 1, "GTVt": 2}
        for modality, voi in product(["PT", "CT"], ["GTVp", "GTVt"]):
            results = self._append_results(
                patient_id,
                self.featureextractor.execute(
                    images[modality],
                    mask,
                    label=vois_label[voi],
                ),
                results,
                modality=modality,
                voi=voi,
            )
        return results


if __name__ == '__main__':
    main()
