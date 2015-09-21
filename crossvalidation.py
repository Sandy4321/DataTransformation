# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.

import pandas
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

# Prints the best scores and the respective parameters used for the ExtraTreesClassifier, BaggingClassifier and GradientBoostingClassifier
def CompareBest(n=5,algorithmlist = ["AdaBoost","Bagging","ExtraTrees","RandomForest","GradientBoosting"]):
    for algorithm in algorithmlist:
        try:
            Scores = pandas.read_csv(str(algorithm) + "ClassifierScores.csv")  
            Scores.drop(['Unnamed: 0'],1,inplace=True)
            Scores = Scores.sort(['AUC'],ascending=False)
            print "\n" + str(algorithm) + "Classifier: \n" + str(Scores.head(n))
        except:
            print "\n" + str(algorithm) + "ClassifierScores.csv does not exist"
    return

# Groups the answers in n equally sized probability groups, given us:
# Ratio of Frauds/Total transactions per group
# Number of transactions per group
# Number of Frauds per group
# Number of NonFrauds per group
# Profit if we were to accept transactions in that group
# The profit function should be a lot better than it is.

def GroupAnswers(clf, test, n=20):
    xtest = test.drop('Fraud',1)
    ytest = test['Fraud'].tolist()
    result = pandas.DataFrame({'Range         ': pandas.Series(map(lambda i: str((i*10000/n)/100.0) + " - " + str(((i+1)*10000/n)/100.0), list(range(0,n,1))),index=range(0,n,1)),'Total': pandas.Series([0]*n,index=range(0,n,1)),'OK': pandas.Series([0]*n,index=range(0,n,1)),'Fraud': pandas.Series([0]*n,index=range(0,n,1)),'Ratio': pandas.Series([0,0],index=[0,1]),'Profit': pandas.Series([0,0],index=[0,1])})
    ygroup = map(lambda x: int(x[1]*n),clf.predict_proba(xtest))
    compare = zip(ytest,ygroup)
    averageprofit = 400
    for i in range(0,n,1):
        result.loc[i,'Fraud'] = compare.count((1,i))
        result.loc[i,'OK'] = compare.count((0,i))
        result.loc[i,'Total'] = compare.count((0,i))+compare.count((1,i))
        result.loc[i,'Profit'] = int(averageprofit* (2*compare.count((0,i)) - 8*compare.count((1,i)))/10)
    result['Ratio'] = 100*result['Fraud']/(result['Total'])
    result['Ratio'] = map(lambda x: str(int(100*x)/100.0) + "%",result['Ratio'])
    result = result[['Range         ','Ratio','Total','Fraud','OK','Profit']]
    print "\n(Profit assumes average 400 GBP Per Purchase)\n"
    print(result.to_csv(sep='\t', index=False))
    return None

#Prints the Confusion Matrix using a certain Treshold given (Default is 0.5), and the profit by deciding purchases on that basis.
def UsingTreshold(clf,test,treshold=0.5):
    xtest = test.drop('Fraud',1)
    ytest = test['Fraud'].tolist()
    result = pandas.DataFrame({'Predict': pandas.Series(["OK","Fraud"],index=[0,1]),'OK': pandas.Series([0,0],index=[0,1]),'Fraud': pandas.Series([0,0],index=[0,1]),'Total': pandas.Series([0,0],index=[0,1])})
    yprob = map(lambda x: x[1],clf.predict_proba(xtest))
    yprob = map(lambda x: 1 if x>treshold else 0,yprob)
    compare = zip(ytest,yprob)
    result.loc[1,'Fraud'] = compare.count((1,1))
    result.loc[1,'OK'] = compare.count((0,1))
    result.loc[1,'Total'] = compare.count((1,1)) + compare.count((0,1))
    result.loc[0,'Fraud'] = compare.count((1,0))
    result.loc[0,'OK'] = compare.count((0,0))
    result.loc[0,'Total'] = compare.count((1,0)) + compare.count((0,0))
    result=result[['Predict','OK','Fraud','Total']]
    print(result.to_csv(sep='\t', index=False))
    averagepurchase = 400
    profit = int(averagepurchase*(compare.count((0,0))*0.2 - compare.count((1,0))*0.8))
    print "\n(Assuming all purchases are worth " + str(averagepurchase) + " GBP)\nprofitapproximation: " + str(profit) + " GBP"
    return

#Computes the AUC for a given model. Not nearly as useful as drawROCCurve
def DecisionTreeAUC(clf,test):
    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']
    return metrics.roc_auc_score(ytest,map(lambda x: x[1],clf.predict_proba(xtest)))

#Draws the ROC Curve for a given model clf
def drawROCCurve(clf,test):
    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']
    yallprob = clf.predict_proba(xtest)
    yprob = map(lambda x: x[1],yallprob)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = metrics.roc_curve(ytest, yprob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(13,13),facecolor="0.99")
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Draws the ROC Curves for the 5 classifiers tested (15 secs to process)
def drawAllROCCurves(test,classifierlist = ['AdaBoost','Bagging','ExtraTrees','GradientBoosting','RandomForest']):
    xtest = test.drop('Fraud',1)
    ytest = test['Fraud']
    plt.figure(figsize=(13,13))
    colors = ['red','lightblue','green','orange','darkblue']
    for i in range(0,5,1):
        clf = pickle.load(open(str(classifierlist[i]) + "Classifier.p","rb"))
        yallprob = clf.predict_proba(xtest)
        yprob = map(lambda x: x[1],yallprob)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = metrics.roc_curve(ytest, yprob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=classifierlist[i] + '(Area = %0.2f)' % roc_auc,color = colors[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Given a model prints the most important n rows and their importance
def featureImportance(clf,test,n=10):
    xtest = test.drop('Fraud',1)
    featureimportance = map(lambda x: int(x*10000)/100.0, clf.feature_importances_)
    featureimportance = zip(featureimportance,xtest.columns.values)
    featureimportance = map(lambda (x,y): [x,y], featureimportance)
    featureimportance = sorted(featureimportance,reverse=True)
    featureimportance = map(lambda x: [str(x[0]) + "%  ",x[1]], featureimportance)
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in featureimportance[:n]]))
    return