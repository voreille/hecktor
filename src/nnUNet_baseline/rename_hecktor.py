from pathlib import Path
import logging
import shutil

import click
from nnunet.dataset_conversion.utils import generate_dataset_json


project_folder = Path(__file__).resolve().parents[2]
# Default paths
path_test = "data/hecktor2021_test"
path_train = "data/hecktor2021_train"
path_data_nnunet = "data/nnUNet_raw_data/Task500_HeadNeckPTCT"



@click.command()
@click.argument("train_folder", type=click.Path(exists=True), default=path_train)
@click.argument("test_folder", type=click.Path(exists=True), default=path_test)
@click.argument("output_folder", type=click.Path(), default=path_data_nnunet)
def main(train_folder, test_folder, output_folder):

    logger = logging.getLogger(__name__)
    logger.info("Starting to write the bb to NIFTI")
    output_folder = project_folder / output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    train_folder = project_folder / train_folder
    test_folder = project_folder / test_folder

    output_images_train_folder = output_folder / "imagesTr"
    output_labels_train_folder = output_folder / "labelsTr"
    output_images_test_folder = output_folder / "imagesTs"

    output_images_train_folder.mkdir(parents=False, exist_ok=True)
    output_labels_train_folder.mkdir(parents=False, exist_ok=True)
    output_images_test_folder.mkdir(parents=False, exist_ok=True)

    logger.info("moving CT train images")
    for file in (train_folder / "hecktor_nii").glob("*_ct.nii.gz"):
        shutil.move(file, output_images_train_folder / file.name.replace("_ct", "_0000"))

    logger.info("moving PT train images")
    for file in (train_folder / "hecktor_nii").glob("*_pt.nii.gz"):
        shutil.move(file, output_images_train_folder / file.name.replace("_pt", "_0001"))

    logger.info("moving labels train images")
    for file in (train_folder / "hecktor_nii").glob("*_gtvt.nii.gz"):
        shutil.move(file, output_labels_train_folder / file.name.replace("_gtvt", ""))

    logger.info("moving CT test images")
    for file in (test_folder / "hecktor_nii").glob("*_ct.nii.gz"):
        shutil.move(file, output_images_test_folder / file.name.replace("_ct", "_0000"))

    logger.info("moving PT test images")
    for file in (test_folder / "hecktor_nii").glob("*_pt.nii.gz"):
        shutil.move(file, output_images_test_folder / file.name.replace("_pt", "_0001"))

    logger.info("Generating .json file")
    generate_dataset_json(str(output_folder / "dataset.json"),
                          str(output_images_train_folder),
                          str(output_images_test_folder),
                          ("CT", "PT"),
                          {0: "background", 1: "GTVt"},
                          "hecktor2021")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
