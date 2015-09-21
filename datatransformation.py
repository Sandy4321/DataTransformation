# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.

import dataimport
import pandas
import datetime
import numpy
import random
import JenksBreaks
import pickle


# Transforms an array as given "lively" into a dataframe similar to the one obtained by the sql query.
# The parameters of this row are defined by the COLUMNS arrays on top.
# The current values are relative to my previous usage, but they'll have to be changed according to what you think is best.
def transformliverequest(row):
    return



# Takes in the raw query. Outputs the query read for training/testing/prediction
# Be careful not to add categorical features with too many options (+20 I would say is too much. Hence no Countries, no Cities, no Boutiques, no Brands,...). You can always have risk groups instead.
def transformquery(df,isTrain=False,one_hot_encode_feature_list = ['PaymentType','category','Newsletter','Currency','FraudStatus','Fallback'],to_delete = ['DateFirstOrder'],boolean_features = ['IsChargeback'],time_differences = [('OrderDate_GMT','DateFirstOrder','TimeSinceFirstOrder')],risk_grouping_features = [('billing_Country',3,True)]):
    df = convertbooleanfeatures(df,boolean_features)
    df = onehotencode(df,isTrain,one_hot_encode_feature_list)
    df = computetimediff(df,time_differences)
    df = deletecolumns(df,to_delete)
    df = riskLevelGrouping(df,isTrain,risk_grouping_features)
    return df


# for each trio [X,Y,Z], df[Z] = df[X]-df[Y]
# features are not deleted here.
def computetimediff(df,timedifferences):
    for (recentdate,olddate,newfeature) in timedifferences:
        df[newfeature] = (df[recentdate] - df[olddate]) / numpy.timedelta64(1, 's')
    return df


#Risk Grouping removes the original feature!
#Each trio in riskgrouping must be of the form (string, int, boolean) where the string is the feature name, the int is the number of the divisions and the boolean defines the existence of an others group which will group cetegories with too few samples.
def riskLevelGrouping(df,isTrain,riskgrouping,limit = 1):
    for (feature,number_of_classes,create_others) in riskgrouping:
        if create_others:
            df = groupsmallocurrences(df,feature,limit)
        if isTrain:
	        catdf = df[feature].value_counts()
	        catdf = {'frequency' : catdf}
	        catdf = pandas.DataFrame(catdf)
	        catdf['FraudRatio'] = catdf.index.map(lambda x: len(df.loc[(df[feature]==x) & (df['Fraud']==1)]) *100.0 / len(df.loc[df[feature]== x]) )
	        breakpoints = JenksBreaks.getJenksBreaks(catdf['FraudRatio'].tolist(),number_of_classes)
	        pickle.dump(dict_to_pickle,open(str(feature) + "_breakpoints.p","wb"))
	    else:
	    	breakpoints = pickle.load(open(str(feature) + "_breakpoints.p","rb"))
        df[feature] = df[feature].apply(lambda x: "RiskLevel " + str(JenksBreaks.classify(catdf.loc[x,'FraudRatio'],breakpoints)))
        df = onehotencode(df,isTrain,[feature])
    return df


# For a given feature in df, changes all values which occur less than limit times to 'Others'. If limit is a float (in [0,1]), then the limit used is len(df)*limit (so values that occur less than limit % timesare changed to 'Others')
def groupsmallocurrences(df,feature,limit = 1):
    catdf = df[feature].value_counts()
    catdf = {'frequency' : catdf}
    catdf = pandas.DataFrame(catdf)
    if type(limit)==float:
        limit = len(df) * limit
    catdf = catdf[catdf['frequency']<=limit]
    catdf = catdf.index
    df[feature] = df[feature].map(lambda x: 'Others' if x in catdf else x)
    return df


# Converts strings ('yes,'no',...) to 1/0
def convertbooleanfeatures(df,booleanfeatures):
    for feature in booleanfeatures:
        df[feature] = df[feature].apply(lambda x: 1.0 if x in ['Yes','yes',1] else (0.0 if x in ['No','no',0] else 0.5))
    return df


# Does one hot encoding of the categorical features
# Deletes original feature
def onehotencode(df,isTrain,categoricalfeatures):
    if isTrain:
        dict_to_pickle = {}
        for f in categoricalfeatures:
            current_categories = set()
            for c in df[f].unique():
                df[f + ": " + str(c)] = (df[f]==c).apply(int)
                current_categories.add(c)
            dict_to_pickle.update({f:current_categories})
            pickle.dump(dict_to_pickle,open("OneHotEncoding_of_" + str(f) + ".p","wb"))
            df.drop(f,1,inplace=True)
    else:
        for f in categoricalfeatures:
            categories = pickle.load(open("OneHotEncoding_of_" + str(f) + ".p","rb"))
            for c in categories:
                df[f + ": " + str(c)] = (df[f]==c).apply(int)
            df.drop(f,1,inplace=True)
    return df



# deletes all columns from featurelist.
def deletecolumns(df,featurelist):
    for feature in featurelist:
        df.drop(feature,1,inplace=True)
    return df



# The dataframe requires a "OrderDate" column with year,month,day,hour,minute,second
# The ordersplit argument decides when to cut the training and testing sets, so that everything after that date is in the testing set.
def splittraintest(df,ordersplit = datetime.datetime.strptime("2014-05-15 01:02:03", "%Y-%m-%d %H:%M:%S"),delete_OrderDate=True):
    train = df[df['OrderDate']<ordersplit].copy(deep=True)
    test = df[df['OrderDate']>=ordersplit].copy(deep=True)
    print "\n***********************"
    print "SPLITTRAINTEST:"
    if delete_OrderDate and ('OrderDate' in df.columns.values):
        train.drop('OrderDate',1,inplace=True)
        test.drop('OrderDate',1,inplace=True)
    if 'OrderDate' not in df.columns.values:
        print "OrderDate not in original set"
    elif delete_OrderDate:
        print "OrderDate deleted"
    else:
        print "OrderDate not deleted"
    print "***********************\n"
    return train,test
 



# Most MachineLearning algorithms work best on balanced sets (same amount of fraud as non-fraud's, for us)
# This function balances the number of train sets as such. If necessary, it can use different ratios by giving the parameter fraudratio ( i.e. fraudratio = 0.6 will give a set with 60% of examples being fraud)
def balanceset(train,fraudratio=0.5):
    train_Fraud = train[train['Fraud']==1]
    train_OK = train[train['Fraud']==0]
    a=len(train_Fraud)*1.0
    b=len(train_OK)*1.0
    if a/(a+b) >= fraudratio:
        a = (b*fraudratio)*1.0/(1-fraudratio)
    else:
        b = a*(1-fraudratio) / fraudratio
    train_Fraud=train_Fraud.ix[random.sample(train_Fraud.index,int(numpy.floor(a)))]
    train_OK=train_OK.ix[random.sample(train_OK.index,int(numpy.floor(b)))]
    train = pandas.concat([train_Fraud,train_OK])
    return train
