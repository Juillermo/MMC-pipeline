"""
Author: Guillermo Romero Moreno (Guillermo.RomeroMoreno@ed.ac.uk)
Date: 29/5/2024

This file contains the basic functions for loading the dataset and obtaining information from it.
"""

import os.path as osp
import time
import json

import numpy as np
import pandas as pd

DATA_PATH = "data/"
PATCOND_VERSION = "5"
CONDS_INFO = None


def load_conditions_info():
    global PATCOND_VERSION
    global CONDS_INFO
    assert PATCOND_VERSION is not None, "Error: you need to initialise the 'PATCOND_VERSION' variable."
    with open(f"{DATA_PATH}conditions_info_v{PATCOND_VERSION}.json", "r") as file:
        CONDS_INFO = json.load(file)


def beautify_name(name, short=False):
    assert CONDS_INFO is not None
    pretty_dict = "short_pretty_dict" if short and name in CONDS_INFO["short_pretty_dict"] else "pretty_dict"
    return CONDS_INFO[pretty_dict][name]


def load_data_and_associations():
    # Load data
    load_conditions_info()
    names = list(CONDS_INFO['pretty_dict'].keys())
    ABC_data = {key: pd.read_csv(f'{DATA_PATH}ABC_data{key}.csv') for key in range(2, 14)}
    
    # Load body-system categories
    groups_dict = CONDS_INFO["groups_dict"]
    bnames = [beautify_name(name, short=True) for name in names]
    assert all(name in CONDS_INFO["pretty_dict"] for name in names)  # This ensures that all names are covered

    # Load forbidden associations and forbidden conditions (by gender)
    forbidden = CONDS_INFO["forbidden_associations"]
    print("Forbidden associations:", forbidden)
    sex_forbidden = CONDS_INFO["sex_forbidden_conditions"]
    print("Forbidden by sex:", sex_forbidden)
    sex_names = {sex: [name for name in names if name not in forbidden_names] for sex, forbidden_names in sex_forbidden.items()}
    
    # Load which conditions are present in each stratum and their prevalence
    with open(f"{DATA_PATH}conditions_in_strata.json", "r") as file:
        conditions_per_stratum = json.load(file)
        conditions_per_stratum = {int(key):val for key, val in conditions_per_stratum.items()}
    conds_prev = {key: pd.read_csv(f'{DATA_PATH}cond_prevs{key}.csv', index_col=0) for key in range(2, 14)}
    
    # Labels
    labels = ['18 $\\leq$ Age $<$ 30 (Men)',
             '18 $\\leq$ Age $<$ 30 (Women)',
             '30 $\\leq$ Age $<$ 40 (Men)',
             '30 $\\leq$ Age $<$ 40 (Women)',
             '40 $\\leq$ Age $<$ 50 (Men)',
             '40 $\\leq$ Age $<$ 50 (Women)',
             '50 $\\leq$ Age $<$ 60 (Men)',
             '50 $\\leq$ Age $<$ 60 (Women)',
             '60 $\\leq$ Age $<$ 70 (Men)',
             '60 $\\leq$ Age $<$ 70 (Women)',
             '70 $\\leq$ Age $<$ 80 (Men)',
             '70 $\\leq$ Age $<$ 80 (Women)',
             '80 $\\leq$ Age $<$ 100 (Men)',
             '80 $\\leq$ Age $<$ 100 (Women)']
    sex_labels = ['Men', 'Women']
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-99']
    
    return {"dats": ABC_data, "conds_per_stratum": conditions_per_stratum,
            "condition_prevalence": conds_prev,
            "names": names, "groups_dict": groups_dict, "bnames": bnames,
            "labels": labels, "age_labels": age_labels, "sex_labels": sex_labels}