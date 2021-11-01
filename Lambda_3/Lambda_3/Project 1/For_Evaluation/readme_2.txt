Team Name: Lambda-3 
Participants:
Tilak Purohit (tilak.purohit@epfl.ch)
Manon Boissa (manon.boissat@epfl.ch)
Xintong Kuang (xintong.kuang@epfl.ch)



Task 1 (Report) :  file name - CS433_ML_Proj_1_Lambda-3.pdf

Task 2 (Codes) : Implementation.py, run.py

6 basic implementations:
File name: Implementation.py  
This file contains the list of functions that were implemented.

- Least squares GD(y, tx, initial w, max iters, gamma): Linear regression using gradient descent
- Least squares SGD(y, tx, initial w, max iters, gamma): Linear regression using stochastic gradient descent
- Least squares(y, tx): Least squares regression using normal equations
- Ridge regression(y, tx, lambda ): Ridge regression using normal equations
- Logistic regression(y, tx, initial w, max iters, gamma): Logistic regression using stochastic/mini-batch gradient descent
- Reg logistic regression(y, tx, lambda, initial w, max iters, gamma): Regularised logistic regression using gradient descent

Note: Implement_help.py contains helper functions for an error-free execution of Implementation.py (keep both these files in the same directory) .  


Project-1 (Higgs-Boson dataset)

Score on aicrowd platform : Categorical accuracy: 79.4   &  F1-score: 68.5


Steps to generate the prediction .csv file (final_output.csv).

1) Create a directory (say ‘dir’), download both the raw data files namely “train.csv” and “test.csv” into the directory. 
2) In the same directory (‘dir’) download these python files - run.py , Project1_helpers.py, My_helpers.py, project1_ridge.py
3) In run.py add the paths for train.csv in variable DATA_TRAIN_PATH, test.csv  in variable DATA_TEST_PATH and, in the variable OUTPUT_PATH add path were you want the generated .csv file to be stored.
4) In python shell execute run.py 

run.py does the data-preprocessing for both test and train set, then use the training set in 4-fold cross-validation manner to train the ridge regression model (our best performing model) where it search for the optimum lambda value and stores the best weight (’w’) parameter. Further the function uses this best ‘w’ for the test data label prediction, and writes the predicted values to  final_output.csv file.  

Note: run.py requires   Project1_helpers.py, My_helpers.py, project1_ridge.py for an error-free execution (keep all these files in the same directory)