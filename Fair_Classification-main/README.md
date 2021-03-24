## Introduction
This folder contains the datasets and source code (i.e. .py files) used in the paper “Unbiased Subdata Selection for Fair Classification: A Unified Framework and Scalable Algorithms.” To successfully run the code in subfolders, you may need to install Python (i.e. .py files).

Below is the detailed explanation of each subfolder.

## Subfolder: Experiment 1
- **Purpose:** Experiment 1 was implemented to test the performance of the algorithm 1 in the paper and compare it with Gurobi.
- **Data Set Info:** The "wine_5000.xlsx" used in Experiment 1 contains 5000 data points with 12 features.
- **Source code Info:** The code folder includes two .py files. "Algorithm1.py" and "Alg1_gurobi.py" are the codes of algorithm 1 and Gurobi, respectively.

## Subfolder: Experiment 2
- **Purpose:** Experiment 2 was implemented to test the performance of the algorithm 2 in the paper and compare it with Gurobi.
- **Data Set Info:** The "wine_55.xlsx" used in Experiment 2 contains 55 data points with 12 features.
- **Source code Info:** The code folder includes two .py files. "IRS.py" and "IRS_gurobi.py" are the codes of algorithm 2 and Gurobi, respectively.

## Subfolder: Experiment 3
- **Purpose:** Experiment 3 was implemented to test the performance of the proposed GSVMF in the paper and compare it with vanilla SVM.
- **Data Set Info:** The "synthetic_data.xlsx" used in Experiment 3 contains 200 data points with 3 features.
- **Source code Info:** The code folder includes two .py files. "GSVMF.py" and "SVM.py" are the codes of the proposed GSVMF and vanilla SVM, respectively.

## Subfolder: Experiment 4
- **Purpose:** Experiment 4 was implemented to test the performance of different classification models. 
- **Data Set Info:** Five files contain datasets used in the proposed GSVMF and GKSVMF, "abalone", "compas", "default", "studentm", and "studentp". Five files contain datasets used in the proposed GLRF, "abalone_LR", "compas_LR", "default_LR", "studentm_LR", and "studentp_LR". The source data are available at http://archive.ics.uci.edu/ml/index.php.
- **Source code Info:** The code folder includes three .py files. "GSVMF_OMR.py", "GKSVMF_OMR.py", and "GLRF_OMR.py" are the codes of GSVMF, GKSVMF, and GLRF with overall misclassification rate fairness, respectively.

## Subfolder: Experiment 5
- **Purpose:** Experiment 5 was implemented to test the performance of the proposed GSVMF and compare it with SSVM proposed by Olfat et al. 2017.  
- **Data Set Info:** Five files contain datasets used in the proposed GSVMF and SSVM, "abalone", "compas", "default", "studentm", and "studentp". The source data are available at http://archive.ics.uci.edu/ml/index.php.
- **Source code Info:** The code folder includes one .py file. "GSVMF_DP.py" is the code of GSVMF with demographic parity fairness. Olfat's implementation is available at https://github.com/molfat66/FairML.

## Subfolder: Experiment 6
- **Purpose:** Experiment 6 was implemented to test the performance of the proposed FCNN in the paper and compare it with vanilla CNN.
- **Data Set Info:** The datasets used in Experiment 6 are available at https://drive.google.com/drive/folders/17KXw2I2gV31LwqB0LAdXGK6DwSYhfVmC?usp=sharing
- **Source code Info:** The code folder includes one .py file. "FCNN.py" is the code of FCNN with overall misclassification rate fairness.

## Subfolder: Experiment 7
- **Purpose:** Experiment 7 was implemented to test the performance of the proposed CNN-F1 for the unbalanced data in the paper and compare it with vanilla CNN.
- **Data Set Info:** The datasets used in Experiment 7 are available at https://drive.google.com/drive/folders/17KXw2I2gV31LwqB0LAdXGK6DwSYhfVmC?usp=sharing
- **Source code Info:** The code folder includes one .py file. "CNNF1.py" is the code of CNN-F1.

Feel free to refer to the data and codes displayed in the subfolders. If used in your research, please cite our paper. 

**References:** Ye, Q. and Xie, W. (2020). Unbiased Subdata Selection for Fair Classification: A Unified Framework and Scalable Algorithms. Available at Optimization Online.
