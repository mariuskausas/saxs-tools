#!/usr/bin/python
d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script allows to a P(r) distribution from a GNOM .out file.

Before use:
- Install pandas, numpy, matplotlib.

Marius Kausas					   2019 12 05

=============================================
"""


import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_gnom(path_to_gnom):
    """ Read GNOM output .out file"""
    
    with open(path_to_gnom) as f:
        read_data = f.readlines()
    f.close()
    
    return read_data


def convert_str_to_float(list_of_str):
    """ Convert a list of str to list of floats."""
    
    list_of_floats = []
    for s in list_of_str:
        list_of_floats.append(float(s))
        
    return list_of_floats


def extract_gnom_pr(gnom):
    """ Extract formated lines of a P(R) distribution from GNOM output .out file."""
    
    # Get the start index of P(R) distribution
    for idx, row in enumerate(gnom):
        if "P(R)" in row:
            pr_idx = idx
            break
            
    # Reformat rows for P(R) distribution
    pr_list = []
    for row in gnom[pr_idx + 2:]:
        pr_list.append(convert_str_to_float(row.split()))
        
    return np.array(pr_list)


def plot_pr_distribution(pr):
    """ Plot P(R) distribution."""
    
    fs = 22
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    
    tick_params = dict(labelsize=fs, length=10, width=1)
    
    r = pr[:,0]
    p = pr[:,1]
    
    ax.plot(r, p/p.max(), linewidth=5, color="tab:blue")
    
    ax.set_xlim(0, r.max() + 10)
    ax.set_ylim(0, 1.1)
    
    ax.tick_params(**tick_params)
    
    plt.tight_layout()
    
    plt.savefig("pr_distribution_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":

    # Argument parser

    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
    argparser.add_argument("-gnom", type=str, help="Path to GNOM .out file", required=True)

    # Parse arguments

    args = argparser.parse_args()
    path_to_gnom = args.gnom

    # Plot P(r) distribution

    gnom = read_gnom(path_to_gnom)
    pr = extract_gnom_pr(gnom)
    plot_pr_distribution(pr)

