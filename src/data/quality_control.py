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
default_output_path = project_dir / f"data/hecktor2022/qc_{center}.csv"


@click.command()
@click.argument('input_folder',
                type=click.Path(),
                default=default_input_folder)
@click.argument('output_path', type=click.Path(), default=default_output_path)
@click.option("--cores", type=click.INT, default=None)
def main(input_folder, output_path, cores):
    input_folder = Path(input_folder)
    vois = [
        f for f in input_folder.rglob("*labels_renamed/*_corrected.nii.gz")
    ]
    patient_ids = set([f.name.split("_")[0] for f in vois])
    # if "mda_test" in center.lower():
    #     patient_ids = [p for p in patient_ids if int(p.split("-")[1]) < 202]
    processor = PatientProcessor(input_folder)
    if cores:
        with Pool(cores) as p:
            result = list(
                tqdm(p.imap(processor, patient_ids), total=len(patient_ids)))
    else:
        result = []
        for patient_id in tqdm(patient_ids):
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
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "patient_id": patient_id,
                        "VOI": voi,
                        "modality": modality,
                        **output_pyradiomics
                    },
                    index=[0],
                )
            ],
            ignore_index=True,
        )
        return results

    def _get_bb(self, mask):
        array = np.transpose(sitk.GetArrayFromImage(mask), (2, 1, 0))
        positions = np.where(array != 0)
        x_min = mask.TransformContinuousIndexToPhysicalPoint([
            np.min(positions[0]).astype(float),
            np.min(positions[1]).astype(float),
            np.min(positions[2]).astype(float),
        ])
        x_max = mask.TransformContinuousIndexToPhysicalPoint([
            np.max(positions[0]).astype(float),
            np.max(positions[1]).astype(float),
            np.max(positions[2]).astype(float),
        ])
        return np.array(
            [x_min[0], x_min[1], x_min[2], x_max[0], x_max[1], x_max[2]])

    def _configure_resampler(self, mask):
        resampling = np.array(mask.GetSpacing())
        bb = self._get_bb(mask)
        size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        self.resampler.SetOutputOrigin(bb[:3])
        self.resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        self.resampler.SetOutputSpacing(mask.GetSpacing())
        self.resampler.SetOutputDirection(mask.GetDirection())
        self.resampler.SetInterpolator(sitk.sitkNearestNeighbor)

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
        try:
            self._configure_resampler(mask)
            mask = self.resampler.Execute(mask)
        except Exception:
            print(
                f"{patient_id} failed to configure resampler, due to direction"
            )
            self.resampler.SetOutputOrigin(mask.GetOrigin())
            self.resampler.SetSize(mask.GetSize())  # sitk is so stupid
            self.resampler.SetOutputSpacing(mask.GetSpacing())
            self.resampler.SetOutputDirection(mask.GetDirection())

        self.resampler.SetInterpolator(sitk.sitkBSpline)
        images = [{
            "image": self.resampler.Execute(sitk.ReadImage(str(f))),
            "modality": f.name.split("__")[1].split(".")[0],
        } for f in image_paths]

        mask_array = sitk.GetArrayFromImage(mask)
        results = pd.DataFrame()
        vois_label = {"GTVp": 1, "GTVn": 2}
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
