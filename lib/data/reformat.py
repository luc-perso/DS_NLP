#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
""" 
    ***
    
    Data preparation: correct CSV files and merge the datasets.
    
    :authors: Elie MAZE, Luc Thomas  

"""


#---------------------------------------------------------------------- MODULES
import os, glob
import pandas as pd
import re
import csv


#-------------------------------------------------------------------- FUNCTIONS
def correctInputFile(path:str, outdir:str)->None:
    """
    Convert multilines comments to single lines to correctly load CSV data into dataframes.
    
    parameters
    ----------
    path : str
           path of the input CSV file.
    
    outdir: str
            path of the output directory.

    return
    ------
    None.
    """
    with open(path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    fname = os.path.basename(path)
    name, ext = os.path.splitext(fname)
    outfile = os.path.join(outdir, name + "_corrected.csv")

    content = ""
    for line in lines:
        line = line.strip()
        if bool(re.match(r'^\d+, ?\"?.', line)):
            content += "\n" + line
        else:
            content += " "+line

    with open(outfile, 'w', encoding="utf-8") as fp:
        fp.write(content)
		
		
def merge(folder:str)->None:
	"""
    Merge all CSV datasets into a single file.
    
    parameters
    ----------
    folder : str
           path of the folder containing all the corrected CSV files.
    
    return
    ------
    None.
    """
	paths = glob.glob(folder+"/*_corrected.csv")
	
	data = pd.DataFrame()
	for path in paths:
		df = pd.read_csv(path, sep=",", engine='python', index_col=0)
		data = pd.concat((data, df))
	
	data = data.reset_index(drop=True)
	outfile = os.path.join(folder, "merged_dataset.csv")
	data.to_csv(outfile, sep=",", encoding="utf-8", quoting=csv.QUOTE_ALL, index=False)
