HECKTOR challenge
==============================

Tools to preprocess the data of the HECKTOR challenge and run a baseline CNN segmentation on the training data (using NiftyNet). This repository contains the code used to prepare the data of the challenge (DICOM to NIFTI, SUV computation and bounding box generation, not needed for the participants). It also contains an example of implementation to resample the data within the bounding boxes and resample back to the original resolution, as well as a NiftyNet baseline implementation to train a CNN on 90% of the training data and evaluate the results on the remaining 10%.


Download Data
------------
To access the data, visit the challenge website: https://www.aicrowd.com/challenges/hecktor and follow the instructions.
The code included here was intended to work with a specific repository structure described in Section Project Organization.
Following `git clone https://github.com/voreille/hecktor.git`, create a `data/` folder in the repository and place the unzipped data in it.

Install Dependencies
------------
To install the necessary dependencies you can use `pip install -r requirements.txt`. It is advised to use it within
a python3 virtual environment.


Resample Data
------------
Run `python src/resampling/cli_resampling.py` to crop and resample the data following the repository structure or use arguments (type `python src/resamping/cli_resampling.py --help` for more informations).

Train a CNN
------------
Use cuda-10 for GPU usage.
`cd src/niftynet` and run `net_segment train -c config3D.ini` for training, followed by
`net_segment inference -c config3D.ini` for inference and `net_segment evaluation -c config3D.ini` for evaluation. 
A random 90%-10% split is used for training and testing. 
Note that the HECKTOR test data will come from a center different from the four training centers, you may want to evaluate your generalization across centers.
A renaming function should be used to comply with the format needed for the test evaluation; run `python rename_output.py` to create a folder with correct file names.

Evaluate Results
------------
An example of how the evaluation will be computed is illustrated in the notebook `notebooks/evaluate_predictions.ipynb`.
For the submission of the test results, you will need to resample back to the original CT resolution. 
This is implemented in this ipynb prior to evaluation and can be used with NiftyNet output or other algorithms' outputs.
Alternatively, run `python src/resampling/cli_get_resolution.py` followed by `python src/resampling/cli_resampling_back.py`
to resample your results back to the original CT resolution. (For more information `python src/resampling/cli_resampling_back.py --help`)

Project OrganizatioN
------------

    ├── README.md                     
    ├── data                           <- NOT in the version control.
    │   ├── resampled                  <- The data in NIFTI resampled and cropped according to the bounding boxes (bbox.csv).
    │   ├── hecktor_nii                <- The data converted in the nifty format with the original geometric frame,
    |   |                                 e.i. the one downloaded form AIcrowd
    │   └── bbox.csv                   <- The bounding box for each patient computed with AUTO_CROP_FUNC
    ├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g.
    │                                    generated with `pip freeze > requirements.txt`
    ├── Makefile                       <- Used to do set up the environment and make the conversion of DICOM to NIFTI.
    ├── notebooks
    │   ├── crop_dataset.ipynb
    │   └── evaluate_predictions.ipynb <- Example of how the evaluation will be computed. This example use the output
    |                                     of the NiftyNet model.
    └── src                            <- Source code for use in this project.
        ├── data                       <- Scripts to download or generate data
        │   ├── __init__.py
        │   ├── bounding_box.py        
        │   ├── dicom_conversion.py    <- Conversion of the DICOM to NIFTI and computation of the bounding boxes.
        │   └── make_dataset.py
        ├── evaluation
        |   ├── __init__.py
        │   ├── cli_compute_scores.py  <- Example of how the score will be computed.
        │   └── scores.py
        ├── niftynet                   <- Code to reproduce the experiments with NiftyNet.
        │   ├── __init__.py
        │   ├── config2D.ini
        │   ├── config3D.ini
        │   ├── dataset_split.csv
        │   └── rename_output.py
        └── resampling                 <- Code to resample the data 
            ├── __init__.py
            ├── cli_get_resolution.py
            ├── cli_resampling_back.py
            ├── cli_resampling.py
            ├── resampling.py
            └── utils.py
         

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
