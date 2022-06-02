from pathlib import Path
from multiprocessing import Pool
from itertools import product

import click
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics.featureextractor import RadiomicsFeatureExtractor

center = "mda_test"
project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data/hecktor2022/processed/"

default_input_folder = data_dir / f"{center}/"
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

    def _append_results(self, *, patient_id, output_pyradiomics, results,
                        modality, voi):
        output_pyradiomics = self._clean_pyradiomics_outptut(
            output_pyradiomics)
        results = results.append(
            {
                "patient_id": patient_id,
                "VOI": voi,
                "modality": modality,
                **output_pyradiomics
            },
            ignore_index=True)
        return results

    def __call__(self, patient_id):
        image_paths = [
            f
            for f in self.input_folder.rglob(f"*images_renamed/{patient_id}*")
        ]
        mask_paths = [
            f
            for f in self.input_folder.rglob(f"*labels_renamed/{patient_id}*")
        ]
        mask = sitk.ReadImage(str(mask_paths[0]))

        self.resampler.SetOutputSpacing(mask.GetSpacing())
        self.resampler.SetOutputDirection(mask.GetDirection())
        self.resampler.SetOutputOrigin(mask.GetOrigin())
        self.resampler.SetSize(mask.GetSize())
        images = [{
            "image": self.resampler.Execute(sitk.ReadImage(str(f))),
            "modality": f.name.split("__")[1]
        } for f in image_paths]

        mask_array = sitk.GetArrayFromImage(mask)
        results = pd.DataFrame()
        vois_label = {"GTVt": 1, "GTVt": 2}
        for voi, image_ in product(vois_label.keys(), images):
            if np.sum(mask_array == vois_label[voi]) == 0:
                continue
            image = image_["image"]
            modality = image_["modality"]
            results = self._append_results(
                patient_id=patient_id,
                output_pyradiomics=self.featureextractor.execute(
                    image,
                    mask,
                    label=vois_label[voi],
                ),
                results=results,
                modality=modality,
                voi=voi,
            )
        return results


if __name__ == '__main__':
    main()
