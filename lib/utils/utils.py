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