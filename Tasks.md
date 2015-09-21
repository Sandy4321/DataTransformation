# What needs to be done:



### Find more algorithms

We still have no algorithms working on categorical data. It would be great if you found (or made yourself :P) some.

** K-Nearest Neighbors ** - Could possibly work if we tune it right? (and it might solve the categorical features issue. See http://scikit-learn.org/stable/modules/neighbors.html#unsupervised-nearest-neighbors )

### Tune Parameters

The current TuneParams functions are relics from my first experiment. They need a lot of work to work. I kept them so that a model somewhat exists. Account for all parameters (don't be lazy!)

### CrossValidation

The most used metric was AUC. If a better one is found please adapt ALL functions in **crossvalidation.py** accordingly.

### GenerateClassifiers

This is mostly done, and easy to change (and add new algorithms). Assuming new algorithms are added, **generateclassifier.py** (and both functions inside it) must be changed. Also it is a good idea to change **CompareBest** and **drawAllROCCurves** in **crossvalidation.py**

### Transform a row into the query form

This is useful for connection with the external world. The row should be minimalist, and we compute as much as we can internally.

### predictFraud

To contact the external world we will use this module. it receives a row, uses **transformliverequest** (from **datatranformations**) to transform it into a raw dataframe form, **transformquery** (from **datatransformations**) to transform it into the "final" dataframe and then uses the classifier (unpickled from the file **modelname.p** )to output a decision probability