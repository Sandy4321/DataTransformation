# -*- coding: utf-8 -*-

import sklearn.ensemble as classifiers
import pandas
from sklearn import metrics
from datatransformation import balanceset, splittraintest


# Tests the efficiency of the GradientBoostingClassifier for the ranges given, saving the results to GradientBoostingClassifierScores.csv
def TuneParamsGradientBoostingClassifier(train,test,fnfrrange=range(1,10,1),nerange=range(120,131,10),lrrange=range(9,12,1),mdrange=range(4,5,1),minauc=0.872,maxauc=0):

    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']
    
    Scores = pandas.read_csv("GradientBoostingClassifierScores.csv")
    Scores.drop(['Unnamed: 0'],1,inplace=True)
    print "\n -- > fnfr / n_estimators / learning_rate / max_depth / AUC"
    print "fnfr range: " + str(fnfrrange)
    print "ne range: " + str(nerange)
    for fnfr in fnfrrange:
        print "fnfr: " + str(fnfr)
        var_fraudnonfraudratio=fnfr/10.0
        temp_train = balanceset(train,var_fraudnonfraudratio)
        xtrain = temp_train.drop('Fraud',1)
        ytrain = temp_train['Fraud']
        for ne in nerange:
            print "ne: " + str(ne)
            var_n_estimators=ne
            for lr in lrrange:
                var_learning_rate=lr*1.0/100
                for md in mdrange:
                    var_max_depth=md
                    gbc = classifiers.GradientBoostingClassifier(n_estimators=var_n_estimators, learning_rate=var_learning_rate, max_depth=var_max_depth)
                    clf = gbc.fit(xtrain, ytrain)
                    auc = metrics.roc_auc_score(ytest,map(lambda x: x[1],clf.predict_proba(xtest)))
                    Scores = Scores.append(pandas.DataFrame({'fnfr': pandas.Series([var_fraudnonfraudratio],index=[0]),'ne': pandas.Series([var_n_estimators],index=[0]),'lr': pandas.Series([var_learning_rate],index=[0]),'md': pandas.Series([var_max_depth],index=[0]),'AUC': pandas.Series([auc],index=[0])}),ignore_index=True)
                    if auc>=minauc:
                        print " -- > " + str(var_fraudnonfraudratio) + " / " + str(var_n_estimators) + " / " + str(var_learning_rate) + " / " + str(var_max_depth) + " / " + str(auc)
                    if auc>=maxauc:
                        maxauc=auc
    print "\n\nBest AUC: " + str(maxauc)
    Scores = Scores.drop_duplicates()
    Scores = Scores.sort(['AUC'],ascending=False)
    Scores.to_csv(path_or_buf="GradientBoostingClassifierScores.csv",sep=',')
    return Scores

# Tests the efficiency of the BaggingClassifier for the ranges given, saving the results to BaggingClassifierScores.csv  
def TuneParamsBaggingClassifier(train,test,fnfrrange=range(8,13,1),nerange=range(140,221,5),msrange=range(30,71,5),minauc=0.852,maxauc=0):

    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']

    Scores = pandas.read_csv("BaggingClassifierScores.csv")  
    Scores.drop(['Unnamed: 0'],1,inplace=True)
    print "\n -- > fnfr / n_estimators / max_samples / AUC"
    print "fnfr range: " + str(fnfrrange)
    print "ne range: " + str(nerange)
    for fnfr in fnfrrange:
        print "fnfr: " + str(fnfr)
        var_fraudnonfraudratio=fnfr/10.0
        temp_train = balanceset(train,var_fraudnonfraudratio)
        xtrain = temp_train.drop('Fraud',1)
        ytrain = temp_train['Fraud']
        for ne in nerange:
            print "ne: " + str(ne)
            var_n_estimators=ne
            for ms in msrange:
                var_max_samples=ms
                bc = classifiers.BaggingClassifier(n_estimators=var_n_estimators, max_samples=var_max_samples, n_jobs=-1)
                clf = bc.fit(xtrain, ytrain)
                auc = metrics.roc_auc_score(ytest,map(lambda x: x[1],clf.predict_proba(xtest)))
                Scores = Scores.append(pandas.DataFrame({'fnfr': pandas.Series([var_fraudnonfraudratio],index=[0]),'ne': pandas.Series([var_n_estimators],index=[0]),'ms': pandas.Series([var_max_samples],index=[0]),'AUC': pandas.Series([auc],index=[0])}),ignore_index=True)
                if auc>=minauc:
                    print " -- > " + str(var_fraudnonfraudratio) + " / " + str(var_n_estimators) + " / "  + str(var_max_samples) + " / " + str(auc)
                if auc>=maxauc:
                    maxauc=auc
    print "\n\nBest AUC: " + str(maxauc)
    Scores = Scores.drop_duplicates()
    Scores = Scores.sort(['AUC'],ascending=False)
    Scores.to_csv(path_or_buf="BaggingClassifierScores.csv",sep=',')
    return Scores

# Tests the efficiency of the ExtraTreesClassifier for the ranges given, saving the results to ExtraTreesClassifierScores.csv
def TuneParamsExtraTreesClassifier(train,test,fnfrrange=range(9,14,1),nerange=range(100,300,10),mssrange=range(10,101,5),critrange=['entropy'],minauc=0.869,maxauc=0):

    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']
    
    Scores = pandas.read_csv("ExtraTreesClassifierScores.csv")  
    Scores.drop(['Unnamed: 0'],1,inplace=True)
    print "\n -- > fnfr / n_estimators / min_samples_split / split_criterion / AUC"
    print "fnfr range: " + str(fnfrrange)
    print "ne range: " + str(nerange)  
    for fnfr in fnfrrange:
        print "fnfr: " + str(fnfr)
        var_fraudnonfraudratio=fnfr/10.0
        temp_train = balanceset(train,var_fraudnonfraudratio)
        xtrain = temp_train.drop('Fraud',1)
        ytrain = temp_train['Fraud']
        for ne in nerange:
            print "ne: " + str(ne)
            var_n_estimators=ne
            for mss in mssrange:
                var_min_samples_split=mss
                for var_criterion in critrange:
                    etc = classifiers.ExtraTreesClassifier(n_estimators=var_n_estimators,criterion = var_criterion,min_samples_split=var_min_samples_split,n_jobs=-1)
                    clf = etc.fit(xtrain, ytrain)
                    auc = metrics.roc_auc_score(ytest,map(lambda x: x[1],clf.predict_proba(xtest)))
                    Scores = Scores.append(pandas.DataFrame({'fnfr': pandas.Series([var_fraudnonfraudratio],index=[0]),'ne': pandas.Series([var_n_estimators],index=[0]),'ms': pandas.Series([var_min_samples_split],index=[0]),'Crit': pandas.Series([var_criterion],index=[0]),'AUC': pandas.Series([auc],index=[0])}),ignore_index=True)
                    if auc>=minauc:
                        print " -- > " + str(var_fraudnonfraudratio) + " / " + str(var_n_estimators) + " / "  + str(var_min_samples_split) + " / " + str(var_criterion) + " / " + str(auc)
                    if auc>=maxauc:
                        maxauc=auc
    print "\n\nBest AUC: " + str(maxauc)
    Scores = Scores.drop_duplicates()
    Scores = Scores.sort(['AUC'],ascending=False)
    Scores.to_csv(path_or_buf="ExtraTreesClassifierScores.csv",sep=',')
    return Scores