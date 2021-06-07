HECKTOR 2021 challenge
==============================

This repository gives an example on how to preprocess the data of the HECKTOR challenge. Any other preprocessing is welcomed and any framework can be used for the challenge, the only requirement is to submit the results in the same coordinate system as the original CT images (same spacing and same origin). This repository also contains the code used to prepare the data of the challenge (DICOM to NIFTI, SUV computation and bounding box generation, not needed for the participants). Moreover, it contains an example of implementation to resample the data within the bounding boxes and resample back to the original resolution.


Download Data
------------
To access the data, visit the challenge website: https://www.aicrowd.com/challenges/miccai-2021-hecktor and follow the instructions.
The code included here was intended to work with a specific repository structure described in Section Project Organization.
Following `git clone https://github.com/voreille/hecktor.git`, create a `data/` folder in the repository and place the unzipped data in it.

Install Dependencies
------------
To install the necessary dependencies you can use `pip install -r requirements.txt`. It is advised to use it within
a python3 virtual environment.

Resample Data
------------
Run `python src/resampling/resample.py` to crop and resample the data following the repository structure or use arguments (type `python src/resamping/cli_resampling.py --help` for more informations).

Evaluate Results
------------
An example of how the evaluation will be computed is illustrated in the notebook `notebooks/evaluate_predictions.ipynb`.
For the submission of the test results, you will need to resample back to the original CT resolution. 
This is implemented in this ipynb prior to evaluation and can be used with NiftyNet output or other algorithms' outputs.
Alternatively, run `python src/resampling/cli_get_resolution.py` followed by `python src/resampling/cli_resampling_back.py`
to resample your results back to the original CT resolution. (For more information `python src/resampling/cli_resampling_back.py --help`)

Project Organization
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
    │   └── evaluate_predictions.ipynb <- Example of how the evaluation will be computed. This example use the output
    |                                     of the NiftyNet model.
    └── src                            <- Source code for use in this project.
        ├── data                       <- Scripts to download or generate data
        │   ├── __init__.py
        │   ├── bounding_box.py        
        │   ├── utils.py               <- Define functions used in make_dataset.py
        │   └── make_dataset.py        <- Conversion of the DICOM to NIFTI and computation of the bounding boxes.
        ├── evaluation
        |   ├── __init__.py
        │   ├── cli_compute_scores.py  <- Example of how the score will be computed.
        │   └── scores.py
        └── resampling                 <- Code to resample the data 
            ├── __init__.py
            └── resample.py
         

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
