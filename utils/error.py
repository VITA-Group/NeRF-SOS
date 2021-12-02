import os, sys
import numpy as np
import torch

DEBUG = True

def CHECK(**kwargs):
    if not DEBUG: return

    for name, value in kwargs.items():
        if torch.isnan(value).any():
            print(f"! [Numerical Error] %s contains nan." % name)
        if torch.isinf(value).any():
            print(f"! [Numerical Error] %s contains inf." % name)
        
def CHECK_ZERO(**kwargs):
    if not DEBUG: return

    for name, value in kwargs.items():
        if (torch.abs(value) <= 1e-12).any():
            print(f"! [Numerical Error] %s contains zeros." % name)

def CHECK_ALL_ZERO(**kwargs):
    if not DEBUG: return

    for name, value in kwargs.items():
        if (torch.abs(value) <= 1e-12).all():
            print(f"! [Numerical Error] %s all zeros." % name)