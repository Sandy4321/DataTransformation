# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.
import pickle

from datatransformation import transformliverequest,transformquery

def predictFraud(row,model="GradientBoostingClassifier.p"):	
    clf = pickle.load(open(model,"rb"))
    return clf.predict_proba(transformquery(transformliverequest(row)))[0][1]