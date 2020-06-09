HECKTOR challenge
==============================

Tools to preprocess the data of the HECKTOR challenge and run a baseline CNN segmentation on the training data (using NiftyNet). This repository contains the code used to prepare the data of the challenge (DICOM to NIfTI, SUV computation and bounding box generation, not needed for the participants). It also contains an example of implementation to resample the data within the bounding boxes and resample back to the original resolution, as well as a NiftyNet baseline implementation to train a CNN on 90% of the training data and evaluate the results on the remaining 10%.


Download Data
------------
To access the data, visit the challenge website: https://www.aicrowd.com/challenges/hecktor and follow the instructions.
The code included here was intended to work with a specific repository structure described in Section Project Organization. 

Resample Data
------------
Run `python src/resamping/cli_resampling.py` to crop and resample the data following the repository structure or use arguments (see documentation of src/resamping/cli_resampling.py).

Train a CNN
------------
`cd src/niftynet` and run `net_segment train -c config3D.ini` for training,
`net_segment inference -c config3D.ini` for inference and `net_segment evaluation -c config3D.ini` for evaluation.

Evaluate Results
------------
An example of how the evaluation will be computed is illustrated in the notebook `notebooks/evaluate_predictions.ipynb`.


Project Organization
------------

    ├── README.md                     
    ├── data
    │   ├── resampled                  <- The data in the nifty resampled and cropped according to the bounding boxes (bb.csv).
    │   ├── processed                  <- The data converted in the nifty format with the original geometric frame
    │   ├── raw                        <- The original dicom data
    │   └── bb.csv                     <- The bounding box for each patient computed with AUTO_CROP_FUNC
    ├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g.
    │                                    generated with `pip freeze > requirements.txt`
    ├── Makefile                       <- Used to do set up the environment and make the conversion of DICOM to NIFTI.
    ├── notebooks
    │   ├── crop_dataset.ipynb
    │   └── evaluate_predictions.ipynb <- Example of how the evalution will be computed. This example use the output of the NiftyNet model.
    └── src                            <- Source code for use in this project.
        │   ├── __init__.py
        │   ├── data                   <- Scripts to download or generate data
        │   ├── bounding_box.py
        │   ├── dicom_conversion.py
        │   └── make_dataset.py
        ├── evaluation
        |   ├── __init__.py
        │   ├── cli_compute_scores.py
        │   └── scores.py
        ├── niftynet
        │   ├── __init__.py
        │   ├── config2D.ini
        │   ├── config2D_vin.ini
        │   ├── config3D.ini
        │   ├── config3D_vin.ini
        │   ├── dataset_split.csv
        │   └── rename_output.py
        ├── resampling
        │   ├── __init__.py
        |   ├── cli_get_resolution.py
        |   ├── cli_resampling_back.py
        |   ├── cli_resampling.py
        |   ├── resampling.py
        |   └── utils.py
        └── tox.ini                       <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
