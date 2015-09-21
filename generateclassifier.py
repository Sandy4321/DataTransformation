# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.

import sklearn.ensemble as classifiers
import pandas
import pickle
from datatransformation import balanceset,transformquery

# Generates the predictor as a pickle file using the algorithm selected. hopefully you'll be able to find algorithms which handle categorical features. I haven't so far.

algorithmlist = ["AdaBoost","Bagging","ExtraTrees","RandomForest","GradientBoosting"]

# generates the classifier with the default parameters (which should be optimal. You'll have to tune the parameters accordingly).
# For an explanation of the variables check  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble  .
# Some time is advised to get used to all the algorithms. Feel free (and indeed do) look for more algorithms to implement!


def GenerateClassifier(algorithm,train,var_n_estimators=130,var_learning_rate=0.1,var_max_depth=4, var_max_samples=1.0,var_criterion='entropy',var_min_samples_split=2):
    xtrain = train.drop('Fraud',1)
    ytrain = train['Fraud']
    if algorithm == "AdaBoost":
        classifier = classifiers.AdaBoostClassifier(n_estimators=var_n_estimators, learning_rate=var_learning_rate)
    elif algorithm == "Bagging":
        classifier = classifiers.BaggingClassifier(n_estimators=var_n_estimators, max_samples=var_max_samples, n_jobs=-1) 
    elif algorithm == "ExtraTrees":
        classifier = classifiers.ExtraTreesClassifier(n_estimators=var_n_estimators,criterion = var_criterion,min_samples_split=var_min_samples_split,n_jobs=-1)
    elif algorithm == "RandomForest":
        classifier = classifiers.RandomForestClassifier(n_estimators=var_n_estimators,criterion=var_criterion,min_samples_split=var_min_samples_split,n_jobs=-1)
    elif algorithm == "GradientBoosting":
        classifier = classifiers.GradientBoostingClassifier(n_estimators=var_n_estimators, learning_rate=var_learning_rate, max_depth=var_max_depth)
    clf = classifier.fit(xtrain, ytrain)
    pickle.dump(clf, open(str(algorithm) + "Classifier.p","wb"))
    print str(algorithm) + "Classifier Generated"
    return clf

# Updates all pickle files with the given train set and the default parameters (which should be optimal)
def GenerateAllClassifiers(train,algorithmlist = ["AdaBoost","Bagging","ExtraTrees","RandomForest","GradientBoosting"]):
    for algorithm in algorithmlist:
        GenerateClassifier(algorithm,train)
    print "All Classifiers Generated"
    return


# Run this to update all predictors, nothing else being required.
# This main() takes about an hour to run, mainly due to getdf() (the other functions take between 2 and 3 minutes to run)