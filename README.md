# INFS7203 Data Mining, Ecoli Project - The University of Queensland

This repo is created for Sem 2, 2022 INFS7203 Data Mining Data-oriented project - Ecoli
Aim for project: To design a classifier with good generalization to differentiate whether a given gene has the function “Cell communication” given the dataset Ecoli.

---

## Conducted by

<ul>
    <a><li>Emilie Aulie <a href="mailto:s4529780@student.uq.edu.au">s4529780@student.uq.edu.au</a> </li>
</ul>

---

## Info about script/directory:

[datasets](./datasets) - the directory where the computed results will be saved
[Ecoli.csv](./datasets/Ecoli.csv) - Provided dataset with labels, for performing training and metrics evaluation.\
The preprocessed dataset may be found in [here](https://drive.google.com/drive/folders/1cnvofUhz84pMR0SztvfOYzIcpq1sR2VT?usp=sharing)\
[Ecoli_test.csv](./data/datasets/Ecoli_test.csv) - Provided test dataset without labels, for performing prediction.\
[Ecoli_train_preprocessed.csv](./datasets/Ecoli_train_preprocessed.csv.) - Preprocessed training dataset\
[Ecolit_test_preprocessed.csv](./datasets/Ecoli_test_preprocessed.csv) - Preprocessed test dataset\
[main.py](./main.py) - Script for running preprocessing of datasets, training the classifier and performing prediction\'
[Result_report.csv](./Result_report.csv) - Test results with predictions on the test data and the accurancy and F1 calculated from training data performed with CV\

---

## Usage

- Language: Python 3.10.2
- Operating system: Windows
- Require packages: `pandas, scikit-learn, numpy, scipy`

To run the script, go into the directory/folder where the folder is placed and perform the command-line:

```shell
python ./main.py
```

You can also run the code directly within the main.py and press button "run", since main is not taking any input parameters.

---
