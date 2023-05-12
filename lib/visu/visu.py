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