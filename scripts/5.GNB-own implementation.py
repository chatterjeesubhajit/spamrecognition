"""
This Gaussian Naive Bayes leaner is broken down into following  parts:

subsetByClass: Create subsets for each class, and calculate prior probablity
meanCal: get all predictors' mean by class
stdCal: get predictors' standard deviation ( assumption : variance is independent of y)
predictPerExample: compute gaussian probabilities for each example
predict: driver predict function to run predictPerExample over all example, this is passed to multithreading
perform_bagging_NB : performs the bagging function on top of single naive bayes implementation
run_naive_bayes : main multithreaded driver function
compute_accuracy : computes the accuracy of the output
"""
import pandas as pd
import numpy as np
import math
import concurrent.futures
import multiprocessing as mp
from sklearn.metrics import confusion_matrix
import sys
# Split the dataset by class values, returns a dictionary

source="/Users/sreekar/Desktop/vectorized-data-master.pkl"
dest="/Users/sreekar/Desktop/naive-bayes-probabilities.csv"

manager = mp.Manager()
Prob=manager.dict()
mean_0 = None
mean_1 = None
prior = None
std_all = None
classPriorP = None
X = None
output_df = pd.DataFrame(columns=["Test Frac", "Train Acc", "Test Acc", "Estimators"])

def subsetByClass(X,classLabel):
    classPriorProb = (X[classLabel].value_counts() / len(X)).to_dict()
    dict_class = dict(iter(X.groupby(classLabel)))
    return classPriorProb,dict_class

def meanCal(dict_class):
    mean_dict={}
    for cls in dict_class.keys():
         mean_dict[cls] = (pd.DataFrame(dict_class[cls].mean(axis=0, skipna=True), columns=['mean'])).T
    return mean_dict

def stdCal(X):
    return (pd.DataFrame(X.std(axis=0, skipna=True), columns=['std'])).T

def predictPerExample(j):
    global mean_0, mean_1, Prob, prior, std_all, X
    Sum=0

    for i in range(0, len(X.columns)):
        if (std_all.iloc[0,i] == 0): continue
        firstpart = (mean_0.iloc[0, i] - mean_1.iloc[0, i]) / std_all.iloc[0, i]
        secondpart = (pow(mean_1.iloc[0, i], 2) - pow(mean_0.iloc[0, i], 2)) / 2 * pow(std_all.iloc[0, i], 2)
        Sum = Sum + firstpart * X.iloc[j, i] + secondpart
    total = 1 + np.exp(prior + Sum)
    Prob[j] = 1 / total

# Make predictions for the examples in the test set 'X'.
# Predictions are made parallely using multiprocessing pool worker.
# Once done all the predictions are in the dictionary names 'Prob'.
# Prob contains the probabilities of the predictions.
# >= 0.5 implies the predicted class is 1
# < 0.5 implies the predicted class is 0
def predict(X,mean_dct,std_all_t):
    global mean_0, mean_1
    global Prob
    global prior, std_all, classPriorP
    Prob.clear()

    std_all = std_all_t
    mean_0 = mean_dct[0]
    mean_1 = mean_dct[1]
    prior = np.log(classPriorP[0] / classPriorP[1]) # Remains same for all
    arr = [j for j in range(0,len(X))]

    with mp.Pool(8) as processes:
        processes.map(predictPerExample, arr)

# Performs bagging of multiple NB classifiers.
# The number of classifiers and the other parameters can be tuned inside this method.
def perform_bagging_NB ():
    global output_df

    df = pd.read_pickle(source)
    df = df.drop(df.columns[0], axis=1)
    for estimators in range(10,11,10):
        try:
            run_naive_bayes (df, estimators)
        except:
            pass

    output_df.to_csv("/Users/sreekar/Desktop/sreekar_output_NB.csv")

# Trains the NB classifier with the given 'data'.
# Total of 'estimators' number of classifiers are trained.
def run_naive_bayes (data, estimators):
    global output_df, Prob, classPriorP, X, std_all
    output = []

    for test_frac in range(10, 26, 5):
        test_df = data.sample(frac=test_frac/100, replace=False).reset_index(drop=True) # Test set
        train_df = data.drop(test_df.index).reset_index(drop=True) # Train set

        # Train the model
        classPriorP, dict_df = subsetByClass(train_df, "Spam-Label")
        mean_dct = meanCal(dict_df)
        std_all = stdCal(train_df.drop(['Spam-Label'], axis=1))

        # Test the model on train set
        Y = train_df['Spam-Label']
        X = train_df.drop(['Spam-Label'], axis=1)
        predict(X, mean_dct, std_all)
        train_acc = compute_accuracy(Y, Prob)

        # Test the model on test set
        Y = test_df['Spam-Label']
        X = test_df.drop(['Spam-Label'], axis=1)
        predict(X, mean_dct, std_all)
        test_acc = compute_accuracy(Y, Prob)

        output.append([round(test_frac/100,2),train_acc,test_acc,estimators])

        local_stdout = sys.stdout
        sys.stdout = open("/Users/sreekar/Desktop/output_NB.txt", "a")
        opt = "{},{},{},{}\n".format(round(test_frac/100,2),train_acc,test_acc,estimators)
        print(opt)
        sys.stdout = local_stdout


    local_out_df = pd.DataFrame(output, columns=["Test Frac", "Train Acc", "Test Acc", "Estimators"])
    output_df.append(local_out_df, ignore_index=True)

# Simply computes the accuracy of a classifier's predictions
# as compared with the true labels.
def compute_accuracy (y_true, y_pred_prob):
    indices = y_true.index
    corr = 0
    for index in indices:
        if((y_true[index] == 1 and y_pred_prob[index] >= 0.5) or (y_true[index] == 0 and y_pred_prob[index] < 0.5 )):
            corr+=1
    return corr/len(indices)

if __name__ == '__main__':
    perform_bagging_NB()

