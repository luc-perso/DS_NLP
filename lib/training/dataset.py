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
import torch
from torch.utils.data import Dataset, IterableDataset


#------------------------------------------------------------------------ CLASS
class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class ZeroShotDataset(IterableDataset):
    def __init__(self, comments):
        self.comments = comments

    def __iter__(self):
        for comment in self.comments:
            yield comment