"""
The Ensembles_DTree has the following main components
1. read_data(...) -> Reads the data with the specified parameters
2. bagging(...) -> Trains n number of decision trees with depth d. Each tree is trained by a random bootstrap sample.
3. predict_test_set(...) -> Predicts the test data using the trained bagging ensembles and returns the predictions array.
4. compute_error(...) -> Computes the error by comparing y_pred and y_true
5. get_confusion_matrix(...) -> Returns the confusion matric using y_pred and y_true
6. print_report(...) -> Prints the results in a user readable format.
7. dectree(...) -> Takes a data frame and trains a decision tree from that.
8. mutual_information(...) -> Computes the MI of an attr-value pair in the train data frame.
9. entropy(...) -> Calculates the entropy of a column.
"""

import concurrent
from datetime import datetime
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import statistics as st
import time
import pandas as pd
import math
# import matplotlib.pyplot as plt
from sklearn import tree as sk_tree
from sklearn.metrics import confusion_matrix
# import graphviz
import sys
import random
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import threading as THRD
import os


# These are the global variables which will be
# used for the sake of Inter Process Communication
# among the pool of processes.
manager = mp.Manager()
mutinfo = manager.dict()
global_df = None
cpu_cores = os.cpu_count()
output_df = pd.DataFrame()

# ==== Project Team ====
# SXS190008 - Sreekar
# SXC190070 - Subhajit
# ======================

# Partition Function:
# -receives the dataframe and creates a dictionary of attribute-value pairs
# - it is called once for creation of starting dictionary which is passed to the ID3 function
def partition(x):
    columns = list(x)
    del (columns[0])
    dict_all = {}
    for c in columns:
        values = list(x[c].unique())
        new_dict = {};
        for i in values:
            new_dict[c, i] = list(x.loc[x[c] == i].index)
            dict_all = {**dict_all, **new_dict}
    return dict_all

# Entropy function:It receives the class variable as input, and does the following checks
# if the size of data =0 (nothing to calculate entropy upon), returns 0
# if proportion of one class in a split subset is 0, return 0 for that entropy ( homogenous class)
# else calculate the entropy  - Summation(Pi*log(Pi) (i=0,1)
# returns value to the mutual_information function
def entropy(y, dist=None):
    if len(y) == 0:
        return 0
    else:
        p0 = p1 = 0
        if dist is None:
            p0 = len([c for c in y if c == 0]) / len(y)
            p1 = 1 - p0
        else:
            df = pd.DataFrame(list(zip(y, dist)),columns=['class', 'dist'])
            s0=df.loc[df['class'] == 0, 'dist'].sum()
            s1=df.loc[df['class'] == 1, 'dist'].sum()
            p0 = s0/(s0+s1)
            p1 = 1 - p0
        if p0 == 0.0 or p0 == 1.0:
            return 0
        else:
            return -((p0 * math.log(p0, 2)) + (p1 * math.log(p1, 2)))

# Mutual information function :
# 1.receives the indices which corresponds to the attribute-value pair presently considered
# 2. receives the present subset of the data, on which entropy and mutual information is to be evaluated
# 3. receives entropy before and after  binary split, calculates weighted entropy after split
# 4. finally subtracts before - weighted entropy to return mutual information
def mutual_information(key_pair, indices, hasDist):

    e_root = 0
    if hasDist:
        e_root = entropy(global_df['class'], dist=global_df["distribution"])
    else :
        e_root = entropy(global_df['class'])

    df_true = global_df[global_df.index.isin(indices)]
    df_false = global_df[~global_df.index.isin(indices)]

    e_true = 0; e_false = 0
    if hasDist:
        e_true = entropy(df_true['class'], dist=df_true["distribution"])
        e_false = entropy(df_false['class'], dist=df_false["distribution"])
    else:
        e_true = entropy(df_true['class'])
        e_false = entropy(df_false['class'])

    e_tot = (len(df_true) / len(global_df)) * e_true + (len(df_false) / len(global_df)) * e_false
    mi = e_root - e_tot
    mutinfo[key_pair] = mi

# dectree: this is main binary decision tree recursion implementation code
# terminal conditions are following ID3 algorithm:
#     -if no splitting attribute left, return mode (max frequency) class the left over data (1 or 0)
#     -if left over data is of only one class, i.e. homogenous , return that class value
#     -if max depth has reached, return mode  class the left over data
#     -if no data is left to be split, return prior to split mode of class ( don't recursively call anymore)
# final output is a nested dictionary representing the decision tree
def dectree(df, dict, depth, hasDist):
    if len(dict.keys()) == 0 or len(df['class'].unique()) == 1 or depth == 0:
        if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
            return 1
        else:
            return 0
    else:
        global mutinfo
        global global_df

        mutinfo.clear()
        global_df = df
        args = []

        for k in dict.keys():
            node = dict[k]
            args.append((k, node, hasDist))
        with mp.Pool(cpu_cores) as processes:
            processes.starmap(mutual_information, args)

        splitnode = ()
        Gain = 0
        for k in mutinfo.keys():
            if (mutinfo[k]) >= Gain:
                Gain = mutinfo[k]
                splitnode = k
        feature = splitnode[0]
        value = splitnode[1]
        dict.pop(splitnode)
        depth -= 1
        if len(df.loc[df[feature] == value]) == 0:
            if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
                return 1
            else:
                return 0
        else:
            left_branch = dectree(df.loc[df[feature] == value], dict, depth, hasDist)  # Recursion Left
        if len(df.loc[df[feature] != value]) == 0:
            if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
                return 1
            else:
                return 0
        else:
            right_branch = dectree(df.loc[df[feature] != value], dict, depth, hasDist)  # Recursion Right
        l_tuple = (feature, value, True)
        r_tuple = (feature, value, False)
        node_dict = {l_tuple: left_branch, r_tuple: right_branch}
        return node_dict

# ID3 function -this is the controlling function for the decision tree
# 1. It receives the data , class/label column , and max depth from user main function
# the depth column is given in skeleton code, however used a bit differently in present context:
#   -as we are not making ID3 recursively execute, we just pass these arguments to our recursive function (dectree)
# this function calls the partition function to create the dictionary of attribute value pairs
# it calls the main decision tree function (dectree) with all required arguments
# receives and returns the decision tree (nested dictionary) created to main function
# "hasDist" will be true only for the boosting algorithm. For the rest, no need to pass this value.
def id3(x, y, attribute_value=None, depth=0, max_depth=1, hasDist=False):
    try:
        x.insert(loc=0, column='class', value=y)
    except:
        pass
    depth = max_depth
    columns = list(x)
    del (columns[0])
    if (hasDist): del (columns[len(columns)-1]) # Delete the distribution column if it exists
    dict_all = {}
    for c in columns:
        col = pd.DataFrame((x[c].unique()))
        dict_all = {}
        if hasDist: dict_all = partition(x.drop("distribution",1))
        else: dict_all = partition(x)
    return dectree(x, dict_all, depth, hasDist)

# predict_example function: It takes in one example at a time and traverses through the binary decision tree to
#                          reach the end node. Based on the end node reached, the mode class of the node is the
#                          predicted class of this example. It returns the predicted class to main function
def predict_example(x, tree):
    all_keys = list(tree.keys())
    attribute = all_keys[0][0]
    value = all_keys[0][1]
    subtree = None
    node = None
    if x[attribute] == value:
        node = (attribute, value, True)
    else:
        node = (attribute, value, False)
    subtree = tree.get(node)
    if subtree == 1:
        return 1
    elif subtree == 0:
        return 0
    else:
        return predict_example(x, subtree)

# compute_error function : takes in predicted and actual class columns and computes the mis-classification proportion
# def compute_error(y_true, y_pred):
#     correct = 0
#     for i in range(len(y_pred)):
#         if y_pred[i] == y_true[i]:
#             correct += 1
#     return (1 - correct / len(y_true))
def compute_error(y_true, y_pred):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    error = np.sum(abs(y_t - y_p)) / len(y_true)
    return error

# visualize function: provided visualization function to display the decision tree in tree structure
def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

# get_confusion_matrix function : returns the cross tab of actual vs predicted class frequencies
def get_confusion_matrix(y_true, y_pred):
    TP = 0;
    TN = 0;
    FN = 0;
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
    return [[TP, FN], [FP, TN]]

# This method takes main training data set and returns the ensemble hypotheses
# as an array. For a given depth "max_depth", "num_trees" number of trees
# are trained and returned.
# "class_column" tells the column index of the class/label
def bagging (x, max_depth, num_trees, class_column=1):
    bootstraps = create_bootstraps(x,num_trees,class_column=class_column)
    all_trees = []
    for bootstrap in bootstraps:
        tree_b = id3(bootstrap[0], bootstrap[1], max_depth=max_depth)
        weight_b = 1 # This is set as 1 because all trees are independent in bagging
        all_trees.append((weight_b, tree_b))
    return all_trees

# This method takes main training data and creates "k" bootstrap samples
# each of size same as that of main training data.
# Returns these bootstraps as an array of tuples.
# Each tuple format is (x_data, y_data)
def create_bootstraps (df, k, class_column=1):
    # Handle the simple errors
    if (class_column < 1):
        print("Invalid class column.")
        return
    if (k < 1):
        print("Invalid number of bootstraps. Must be >= 1.")
        return

    # Create k bootstrap samples
    bootstraps = [] # Empty bins
    class_column-=1
    for i in range(0, k):
        df_k = df.sample(frac=1, replace=True)
        y_k = df_k[class_column]
        x_k = df_k.drop(class_column, 1)
        x_k = x_k.sample(n=250, axis=1)
        bootstraps.append((x_k, y_k))

    return bootstraps

# This method accepts train set and num of ensemble trees
# to train. Returns an array of ensembles [(alpha_i, hyp_i)]
def boosting (data, max_depth, num_stumps, class_column=1):
    y = data[class_column-1]
    x = data.drop(class_column-1, 1)
    sample_size = len(list(x.index))
    d_i = [1/sample_size]*sample_size # The initial distribution

    ensembles = [] # Array of ensembles [(alpha_i, hyp_i)]
    for i in range(0,num_stumps):
        x.insert(loc=len(x.columns), column="distribution", value=d_i)
        h_i = id3(x,y,max_depth=max_depth, hasDist=True) # ith decision tree
        d_i = list(x["distribution"])
        del x["distribution"]
        y_pred = predict_test_set(x, type="tree", h_tree=h_i)
        err_i = round(compute_error_boosting(y, y_pred, d_i),3) # error of ith decision tree
        alpha_i = get_hypothesis_weight(err_i) # weight of ith decision tree
        d_i = get_new_distribution(d_i, alpha_i, y, y_pred) # new distribution for next dtree
        ensembles.append((alpha_i, h_i))
    return ensembles

# Computes the errors that are made in the current iteration.
# These errors are used to change the weights on misclassifications
# in the next iterations.
def compute_error_boosting (y_true, y_pred, d_i):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    error = np.sum(abs(y_t - y_p) * d_i) / sum(d_i)
    return error

# This method takes error and returns the alpha value
def get_hypothesis_weight(error):
    a = (1-error)/error
    return 0.5*math.log(a)

# Computes the new distribution for the training set
# based on the current_dis and the current predictions made.
def get_new_distribution(prev_dis, alpha, y_true, y_pred):
    y_t=np.array(y_true)
    y_p = np.array(y_pred)
    p_dis = np.array(prev_dis)
    n_dis=p_dis*np.exp((-1 + 2* abs(y_t-y_p))*alpha)
    new_dis=n_dis.tolist()
    return new_dis

# Takes the test set and computes the prediction.
# Returns the array of BEST predictions as combined
# by the ensemble hypothesis.
# For ensemble models, "h_ens" parameter must be passed
# For a single tree model, "h_tree" parameter must be passed.
def predict_test_set(test_x, type, h_ens=[], h_tree=None):
    if type != 'tree' and type != 'bagging_tree' and type != 'boosting_tree':
        print("Provide the type of model - 'tree', 'bagging_tree', or 'boosting_tree'")
        return
    num_of_examples = len(list(test_x.index))
    predictions = []
    for i in range(0, num_of_examples):
        preds_i = []
        for h in h_ens:
            pred = predict_example(test_x.iloc[i], h[1])
            if (type == "boosting_tree" and pred == 0): preds_i.append(h[0] * -1)
            else: preds_i.append(h[0] * pred)
        if (type == "bagging_tree"):
            try :predictions.append(st.mode(preds_i)) # Final prediction of bagging
            except: predictions.append(1) # Tie breaking
        elif (type == "boosting_tree"):
            if sum(preds_i) > 0: predictions.append(1)
            else: predictions.append(0)
        elif (type == "tree"): predictions.append(predict_example(test_x.iloc[i], h_tree)) # Prediction using simple tree
    return predictions

# The main execution of the assignment begins here
def own_bagging(depths=[], trees=[]):
    global output_df
    data = read_data("vectorized-data-master", data_class_column=1)
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    start = datetime.now()
    for depth in depths:
        for tree_len in trees:
            # Send the training data to the bagging algorithm
            all_trees = bagging(train_df, class_column=1, max_depth=depth, num_trees=tree_len)
            # Predict the test set with all the trees
            predictions = predict_test_set(test_x, type="bagging_tree", h_ens=all_trees)
            print_report(predictions, test_y, depth=depth, trees=tree_len)
            output_df["D-{}-Pred".format(depth)] = predictions
            output_df["D-{}-True".format(depth)] = test_y

    output_df.to_csv("/Users/sreekar/Desktop/Graphs/sreekar_output_RF.csv")
    end = datetime.now()

# The main execution of the assignment begins here
def own_boosting(depths=[], trees=[]):
    data_set_name = "mushroom"
    data_columns_to_drop = []
    data_class_column = 1

    # Read the data files
    train_data_path = "./data/{}.train".format(data_set_name)
    test_data_path = "./data/{}.test".format(data_set_name)
    train_df = pd.read_csv(train_data_path, delimiter=",", header=None)
    test_df = pd.read_csv(test_data_path, delimiter=",", header=None)
    test_y = list(test_df[data_class_column-1])
    del test_df[data_class_column - 1]

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c - 1]
        del test_df[c - 1]

    for depth in depths:
        for tree in trees:
            # Boosting algorithm
            all_trees = boosting(train_df, depth, tree, data_class_column)

            # Predict the test set with all the trees
            predictions = predict_test_set(test_df, type="boosting_tree", h_ens=all_trees)

            # Compute the error and accuracy
            print_report(predictions, test_y, depth=depth, trees=tree)

# Scikit learn bagging
def scikit_bagging(depths=[], trees=[]):
    data = read_data("vectorized-data-master", data_class_column=2)
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    for d in depths:
        for t in trees:
            dtree = DecisionTreeClassifier(max_depth=d)
            clf = BaggingClassifier(base_estimator= dtree,n_estimators=t, bootstrap=True).fit(train_df.drop(1,1), train_df[1])
            pred = clf.predict(test_x)
            print_report(pred, test_y, depth=d, trees=t)

# Scikit learn boosting
def scikit_boosting(depths=[], stumps=[]):
    data = read_data("mushroom")
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    for d in depths:
        for s in stumps:
            dtree = DecisionTreeClassifier(max_depth=d)
            clf = AdaBoostClassifier(base_estimator=dtree, n_estimators=s).fit(train_df.drop(0,1), train_df[0])
            pred = clf.predict(test_x)
            print_report(pred, test_y, depth=d, trees=s)

# Simply prints the results in a formatted way on to the console.
# If you desire to write the result to a file, uncomments the first 2 lines
# and the last 1 line in the below method and pass the file's path in line 2.
def print_report(y_pred, y_true, depth=0, trees=1):
    # stdout_local = sys.stdout
    # sys.stdout = open("/Users/sreekar/Desktop/dec-tree-bagging", "a")
    err = compute_error(y_true, y_pred)
    acc = 1 - err
    print("========================= Depth: {} and Trees: {} ==============================".format(depth, trees))
    print("Error   : ", round(err * 100, 2))
    print("Accuracy: ", round(acc * 100, 2))
    print("C Matrix: \n", confusion_matrix(y_true, y_pred))
    print("========================= *********************** ==============================".format(depth, trees))
    # sys.stdout = stdout_local

# Reads the data (train and test) with name "data_set_name".
# Train Data Path: <user-directory>/data_set_name.train
# Test Data Path: <user-directory>/data_set_name.test
# Any unwanted columns can be dropped from the data set as indicated
# in "data_columns_to_drop" array. Default is empty
def read_data(data_set_name, data_class_column=1, data_columns_to_drop=[]):
    # Read the data files
    train_data_path = "/Users/sreekar/Desktop/{}.train".format(data_set_name)
    test_data_path = "/Users/sreekar/Desktop/{}.test".format(data_set_name)
    train_df = pd.read_pickle(train_data_path)
    test_df = pd.read_pickle(test_data_path)

    test_y = list(test_df[data_class_column-1])
    del test_df[data_class_column - 1]

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c - 1]
        del test_df[c - 1]

    return (train_df, test_df, test_y)

if __name__ == "__main__":
    print("Running model on {} processor cores.".format(cpu_cores))
    own_bagging(depths=[5,10,15,20],trees=[10])
    # own_boosting([1,2], [20,40])
    # scikit_bagging([3,5], [10,20])
    # scikit_boosting([1,2], [20,40])

