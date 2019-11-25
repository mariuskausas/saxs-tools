d = """
=============================================
Analysis of small-angle X-ray scattering data.

The script provides a way to perform CORMAP analysis for a set of scattering profiles.

Reference: Franke, D., et al, (2015), Nature Methods, 12, 419-422.

The scripts does the following:
(i) Load a set of .dat files.
(ii) Calculates CORMAP and P values for each unique combination of .dat files.
(iii) Adjusts P-values using Bonferroni correction.
(iv) Generates two matrices with values of CORMAP and adjusted P values
filled only in the upper right triangle.
(v) Produces heatmap of CORMAP and adjusted P-values. 

Before use:
- Install numpy, statsmodel, matplotlib and seaborn.

Make sure you have datcmp in your path. Please refer to ATSAS installation.

To use the script provide:
- Path to a set of experimental scattering curve files (.dat) together with 3 columns
of q, I(q) and sigma values.

Marius Kausas					   2019 11 24
=============================================
"""


import glob
import subprocess
from itertools import combinations
import numpy as np
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_dats(path_to_dats):
    """ Prepare a list of .dat files"""
    
    dats = glob.glob(path_to_dats)
    dats.sort()
    
    return dats


def run_cormap(dats):
    """ Run DATCMP command with a CORMAP test for a set of .dat scattering files."""
    
    # Run CORMAP with Bonferroni adjustment
    command = subprocess.run(["datcmp", "--test", "CORMAP", "--adjust", "FWER"] + dats, stdout=subprocess.PIPE)
    
    # Get the value row
    all_output = command.stdout.decode("utf-8").split("\n ")
    value_row = all_output[2]
    
    # Get CORMAP and P values
    c = value_row.split()[-3]
    p = value_row.split()[-2]
    
    return c, p


def bonferroni(p_values):
    """ Perfrom Bonferroni P-value adjustment."""
    
    p_adj_values = multitest.multipletests(p_values, alpha=0.05, method="bonferroni")[1]
    
    return p_adj_values


def populate_mat(shape, value_l):
    """ From a list of values, generate a matrix with values only in upper triangle."""
    
    # Generate a matrix
    mat = np.zeros((shape, shape))
    
    # Set slices for keeping track of upper triangle
    slice_i = 0
    slice_j = shape - 1
    
    # Populate upper triangle with values from a list
    for i in range(shape - 1):
        mat[i:i + 1, i + 1:] = np.array(value_l[slice_i: slice_j])
        slice_i = slice_j
        slice_j = slice_i + shape - 2 - i
        
    return mat


def cormap(path_to_dats):
    """ Perform CORMAP test for a given set of .dat scatterig files."""
    
    # Prepare .dat files
    dats = prepare_dats(path_to_dats)
    n_dats = len(dats)
    
    # Define combinations
    dats_combs = list(combinations(dats, 2))
    
    # Run CORMAP for all combinations
    cormap_l = [run_cormap([comb[0], comb[1]]) for comb in dats_combs]
    
    # Create lists with CORMAP and P values
    c_l = [float(value[0]) for value in cormap_l]
    p_l = [float(value[1]) for value in cormap_l]
    
    # Bonferroni adjustment
    p_ajd_l = bonferroni(p_l)
    
    # Generate CORMAP and P-adjusted value matrices (only the upper triangle)
    c_mat = populate_mat(n_dats, c_l)
    p_adj_mat = populate_mat(n_dats, p_ajd_l)
    
    return c_mat, p_adj_mat


def plot_c_mat(c_mat, output_name, annot=False):
    """ Plot adjusted CORMAP value matrix."""
    
    shape = c_mat.shape[0]
    plt.figure(figsize=[12, 10])
    sns.heatmap(c_mat, annot=annot, fmt='.1f', xticklabels=np.arange(1, shape + 1), yticklabels=np.arange(1, shape + 1))
    plt.savefig(output_name + "c_mat.png", dpi=300)
    plt.close()
    
    
def plot_p_adj_mat(p_adj_mat, output_name, annot=False):
    """ Plot adjusted P-value matrix."""

    shape = p_adj_mat.shape[0]
    plt.figure(figsize=[12, 10])
    sns.heatmap(p_adj_mat, annot=annot, fmt='.5f', vmin=0, vmax=1, xticklabels=np.arange(1, shape + 1), yticklabels=np.arange(1, shape + 1))
    plt.savefig(output_name + "_p_adj_mat.png", dpi=300)
    plt.close()
    
