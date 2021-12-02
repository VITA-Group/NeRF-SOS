
import os, sys
import math, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import imageio
import json

import configargparse

def read_config_file(file_path):
    config_file = configargparse.DefaultConfigFileParser()
    with open(file_path, 'r') as f:
        args = config_file.parse(f)
        print(args)
    return args

def update_arg_config(args, config_file, keys=[]):
    if len(keys) > 0:
        keys = set(vars(args)).intersection(set(keys))
    else:
        keys = set(vars(args))

    for name in keys:
        if name in config_file:
            setattr(args, name, config_file[name])

# Compare args1 with args2
def compare_args(args1, args2, keys=[]):

    if len(keys) == 0:
        keys = vars(args2).keys()

    same = True
    for k in keys:
        if not hasattr(args1, k) or getattr(args1, k) != getattr(args2, k):
            print(k)
            same = False
            break
            # setattr(args, name, config_file[name])
    return same

# Update args1 with args2
def update_args(args1, args2, keys=[]):

    if len(keys) == 0:
        keys = vars(args2).keys()

    for k in keys:
        if hasattr(args1, k):
            setattr(args1, k, getattr(args2, k))
    return args1