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
import re


#-------------------------------------------------------------------- FUNCTIONS
def findCAPSLOCK(comment):
    r = re.compile(r"[A-Z]")
    capslock = r.findall(comment)
    return len(capslock)

def find_chain_CAPSLOCK(comment):
    r = re.compile(r"[A-Z]{2,}")
    capslock = r.findall(comment)
    return len(capslock)

def find_exclamation(comment):
    r = re.compile(r"\!")
    exclamation = r.findall(comment)
    return len(exclamation)

def find_chain_exclamation(comment):
    r = re.compile(r"(\! ){2,}")
    exclamation = r.findall(comment)
    return len(exclamation)

def find_interogation(comment):
    r = re.compile(r"\?")
    interogation = r.findall(comment)
    return len(interogation)

def find_chain_interogation(comment):
    r = re.compile(r"(\? ){2,}")
    interogation = r.findall(comment)
    return len(interogation)

def find_etc(comment):
    r = re.compile(r"(\. ){2,}")
    etc = r.findall(comment)
    return len(etc)