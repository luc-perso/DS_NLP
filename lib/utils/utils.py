#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
""" 
    ***
    
    Data (pre)processing.
    
    :authors: Elie MAZE, Luc Thomas  

"""


#---------------------------------------------------------------------- MODULES
import numpy as np
import pandas as pd


#-------------------------------------------------------------------- FUNCTIONS
def getErrors(df, y_true, y_pred, col_x):
    mask_errors = (np.array(y_true)!=np.array(y_pred))
    errors = df[mask_errors]

    errors = pd.DataFrame(errors[col_x].value_counts())
    values = pd.DataFrame(df[col_x].value_counts())

    errors = errors.merge(values, left_index=True, right_index=True)
    errors.columns = ["errors", "total"]

    errors["error_rate"] = errors["errors"] / errors["total"]
    return errors.sort_index()

def getZeroShotPredictions(predictions, index):
    labels = predictions.at[index, "labels"]
    scores = predictions.at[index, "scores"]
    if isinstance(labels, str):
        labels = eval(eval)
        scores = eval(scores)

    indexes = np.argsort(scores)[::-1]

    i=0
    res = pd.DataFrame(columns=["labels", "scores"])
    for idx in indexes:
        res.loc[i] = [labels[idx], scores[idx]]
        i+=1

    return res

def showReview(review, wrap=20):
    fields = review.split(" ")
    nb_fields = len(fields)
    if nb_fields < wrap:
        print(review)
        return
    
    for i in range(0,nb_fields-wrap, wrap):
        print(" ".join(fields[i:i+wrap]))