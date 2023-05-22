#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
""" 
    ***
    
    DL model Explicability.
    
    :authors: Elie MAZE, Luc Thomas  

"""


#---------------------------------------------------------------------- MODULES
import torch
from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients


#-------------------------------------------------------------------- FUNCTIONS
def construct_input_and_baseline(text, tokenizer, max_length=128):

    baseline_token_id = tokenizer.pad_token_id 
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id 

    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
   
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    token_list = tokenizer.convert_ids_to_tokens(input_ids)

    baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device='cpu'), torch.tensor([baseline_input_ids], device='cpu'), token_list

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
       
def explain(model, tokenizer, text, true_class, mapping_classes):
    def model_output(inputs):
        return model(inputs)[0]

    model_input = model.base_model.embeddings
    lig = LayerIntegratedGradients(model_output, model_input)
    
    input_ids, baseline_input_ids, all_tokens = construct_input_and_baseline(text, tokenizer)
    
    for target_class in mapping_classes.keys():
        print()
        print("CLASS: "+str(mapping_classes[target_class]), end="\n\n")
        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=baseline_input_ids,
                                            target=target_class,
                                            return_convergence_delta=True,
                                            internal_batch_size=1
                                            )

        attributions_sum = summarize_attributions(attributions)

        score_vis = viz.VisualizationDataRecord(
                                word_attributions = attributions_sum,
                                pred_prob = torch.max(model_output(input_ids)),
                                pred_class = torch.argmax(model_output(input_ids)).numpy(),
                                true_class = true_class,
                                attr_class = text,
                                attr_score = attributions_sum.sum(),       
                                raw_input_ids = all_tokens,
                                convergence_score = delta)

        viz.visualize_text([score_vis])