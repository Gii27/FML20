# FML HW3
Net ID: ls4330

Boosting.py is the python script for question B(i) in HW3.

It takes abalone.data as input, reformat it and take the first 3133 examples for training and rest for testing.
Uses the mean of each feature as threshold for weak learners.

data_arrange(filename) did the reformat and get stumps which are means of each feature and output df[i, j] = y_ih_j(x_i)
adaBoost and logistic_boost are the boosting algorithm that returns the final alpha vector. 
boosting_test is function for testing with stumps and alpha as input.