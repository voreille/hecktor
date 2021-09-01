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
Run `python src/resampling/resample.py` to crop and resample the data following the repository structure or use arguments (type `python src/resamping/resample.py --help` for more informations).

Evaluate Results
------------
An example of how the segmentation (task 1) will be evaluated is illustrated in the notebook `notebooks/evaluate_segmentation.ipynb`. Note that the Hausdorff distance at 95 % implemented in https://github.com/deepmind/surface-distance will be used in the challenge (not the one found in `surc/evaluation/scores.py`).
The concordance index used to evaluate task 2 and 3 is impemented in the function `concordance_index(event_times, predicted_scores, event_observed=None)` from the file `src/aicrowd_evaluator/survival_metrics.py`.

Submission
------------
Dummy examples of correct submission for task 1 and 2 can be found in `notebooks/example_seg_submission.ipynb` and `notebooks/example_surv_submission.ipynb`respectively.


Project Organization
------------

    ├── README.md                     
    ├── data                              <- NOT in the version control
    │   ├── resampled                     <- The data in NIFTI resampled and cropped according to the bounding boxes (bbox.csv).
    │   ├── hecktor_nii                   <- The data converted in the nifty format with the original geometric frame,
    |   |                                    e.i. the one downloaded form AIcrowd
    │   └── bbox.csv                      <- The bounding box for each patient computed with bbox_auto function from src.data.bounding_box
    ├── requirements.txt                  <- The requirements file for reproducing the analysis environment, e.g.
    │                                        generated with `pip freeze > requirements.txt`
    ├── Makefile                          <- Used to do set up the environment and make the conversion of DICOM to NIFTI
    ├── notebooks
    |   ├── example_seg_submission.ipynb  <- Example of a correct submission for the segmentation task (task 1).
    |   ├── example_surv_submission.ipynb <- Example of a correct submission for the survival task (task 2).
    │   └── evaluate_segmentation.ipynb   <- Example of how the evaluation will be computed.
    └── src                               <- Source code for use in this project
        ├── aicrowd_evaluator             <- Source code for the evaluation on the AIcrowd platform
        │   ├── __init__.py
        │   ├── surface-distance/         <- code to compute the robust Hausdorff distance availabe at https://github.com/deepmind/surface-distance        
        │   ├── evaluator.py              <- Define the evaluator class for task 1 and 2
        │   ├── segmentation_metrics.py   <- Define the metrics used in the segmentation task.
        |   ├── requirements.txt          <- The requirements file specific to this submodule
        │   └── survival_metrics.py       <- Define the metrics used for the survival task.
        ├── data                          <- Scripts to generate data
        │   ├── __init__.py
        │   ├── bounding_box.py        
        │   ├── utils.py                  <- Define functions used in make_dataset.py
        │   └── make_dataset.py           <- Conversion of the DICOM to NIFTI and computation of the bounding boxes
        ├── evaluation
        |   ├── __init__.py
        │   └── scores.py                 <- (DEPRECATED) used to illustrate how the segmentation is evaluated. Refer to `src/aicrowd_evaluator`
        |                                    submodule for the actual evaluation of the challenge.
        └── resampling                    <- Code to resample the data 
            ├── __init__.py
            └── resample.py
         

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
