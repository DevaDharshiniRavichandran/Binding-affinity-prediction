Overview

This repository contains code and resources for training and evaluating machine learning models designed to work with specific datasets. The project provides a set of models derived from a base model (bap.py) and includes variations with different configurations and training datasets. Results from the models are saved in the results folder for easy reference.

File Descriptions


onefile.py:
Base model derived from bap.py.
Trains the model using the dataset specified in the tcr file.


onefile_mod.py:
Modified version of the base model training.
Uses the same dataset (tcr file) as onefile.py.


twofile.py:
Base model derived from bap.py.
Trains the model using the dataset specified in the epi file.


twofile_mod.py:
Modified version of the base model training.
Uses the same dataset (epi file) as twofile.py.


onefilev2.py:
Variation of onefile.py with a different learning rate of 0.01 (instead of 0.001 in the original model).


twofilev2.py:
Variation of twofile.py with a different learning rate of 0.01 (instead of 0.001 in the original model).


Setup and Execution

Environment Setup

Install Python 3.10.

The code and required libraries have been tested and confirmed to work with this version.

Install all required libraries.


Check the import statements in the files to identify dependencies.

No specific versioning is required; the libraries are compatible with Python 3.10.


Adjust File Paths

Before running the code, ensure the file paths for the datasets are updated to match your directory structure. Each script contains a section where dataset paths are defined. Modify these paths to point to your local files.


Notes

Model Variations: Each script reflects specific changes or configurations to the base model (bap.py), such as learning rate or dataset used. Ensure you understand the purpose of each script to select the one that suits your needs.

Results Folder: All outputs are automatically saved to the results folder for organized access.


Learning Rate Changes:

onefilev2.py and twofilev2.py use a learning rate of 0.01.

Other scripts use the default learning rate of 0.001.
