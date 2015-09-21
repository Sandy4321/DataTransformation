# -*- coding: utf-8 -*-
# Contact nuno.carneiro.farfetch.com for errors or if explanations are needed.

# This must be run frequently (every month, or maybe every 2 weeks) to train the predictors on the new data.
# It assumes that all the default values (in each module) are optimal

from importdata import getdf
from datatransformation import balanceset, transformquery
from generatepredictor import GenerateAllClassifiers

def main():
    train = balanceset(transformquery(getdf()))
    GenerateAllClassifiers(train)
    return