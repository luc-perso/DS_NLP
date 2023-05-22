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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import plasma, d3, Turbo256
from bokeh.plotting import figure
from bokeh.transform import transform
import bokeh.io
import bokeh.plotting as bpl
import bokeh.models as bmo

from sklearn import manifold


#------------------------------------------------------------- STATIC VARIABLES
MARKERS = ['square', 'circle', 'asterisk', 'triangle', 'diamond']


#-------------------------------------------------------------------- FUNCTIONS
def plotCMX(labels, cm):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    ax.set_xticklabels([""]+labels)
    plt.ylabel('Actuals', fontsize=18)
    ax.set_yticklabels([""]+labels)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def plotErrors(errors, col_x):
    errors = pd.DataFrame(errors[ ["errors", "total"]].stack()).reset_index(drop=False)
    errors.columns = ["company", "attribute", "count"]

    sns.catplot(data=errors, x="company", y="count", hue="attribute", kind="bar")
    plt.title("Distributions of errors per "+col_x)
    plt.show()

def plot_learning_curves(num_epochs, train_loss, val_loss, train_accuracy, val_accuracy, figsize=(12, 5)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title(f"Learning curves over {num_epochs:d} epochs")

    ax[1].plot(train_accuracy, label='Train accuracy')
    ax[1].plot(val_accuracy, label='Validation accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].set_title(f"Accuracy score over {num_epochs:d} epochs")

    plt.show()

def plotVectors(vectors, descriptions, serie_values, hue_value, legend, markers=MARKERS, figsize=(900, 600), method="tsne"):

    list_x, list_y = [], []
    if method=="tsne":
        tsne_data = manifold.TSNE(n_components=2).fit_transform(vectors)
        list_x = tsne_data[:,0]
        list_y = tsne_data[:,1]
        
    if list_x.size<=0 or list_y.size<=0:
        return
    

    color_mapper = LinearColorMapper(palette='Plasma256', low=min(hue_value), high=max(hue_value))

    source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=descriptions, targets=hue_value, dset=serie_values))
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ('desc', '@desc'),
        ('targets', '@targets'),
        ('dset', '@dst')
    ])

    p = figure(frame_width=figsize[0], frame_height=figsize[1], tools=[hover], title=method.upper()+" applied on input vectors")

    p.scatter('x', 'y', size=10, source=source, legend_field='dset', 
              color={'field': 'targets', 'transform': color_mapper},
              marker=bokeh.transform.factor_mark('dset', markers, legend)
    )

    bpl.show(p)
        
    