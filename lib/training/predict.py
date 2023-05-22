#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
""" 
    ***
    
    Inference.
    
    :authors: Elie MAZE, Luc Thomas  

"""


#---------------------------------------------------------------------- MODULES
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


#------------------------------------------------------------------------ CLASS
class PredictCommentDataset(Dataset):
    def __init__(self, comments, tokenizer):
        self.comments = comments
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])

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
            "attention_mask": encoding["attention_mask"].flatten()
        }


#-------------------------------------------------------------------- FUNCTIONS
def encode_inputs(tokenizer, texts, max_length=128):
    batch_size = min(len(texts), 32)

    # Tokenization
    encoded_inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    return input_ids, attention_mask
    
def getOutputs(model, input_ids, attention_mask):
    outputs = None
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs

def getEmbeddings(model, input_ids, attention_mask, mean=False):
    embeddings = None
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        
        if mean:
            embeddings = embeddings.mean(axis=1)
            
    return embeddings


def getTextsEmbeddings(comments, tokenizer, model, device, batch_size=32, mean=False):
    dataset = PredictCommentDataset(comments, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state
            if mean:
                embedding = embedding.mean(axis=1)
            embeddings += [embedding.cpu().detach().numpy()]

    embeddings = np.concatenate(embeddings, axis=0)
    
    return embeddings