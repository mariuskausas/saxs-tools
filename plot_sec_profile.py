#!/usr/bin/python
d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script allows ploting size exclusion chromatogram of a SEC-SAXS run.

The input file is provided as a .csv file.

Before use:
- Install pandas, numpy, matplotlib.

Marius Kausas					   2019 12 05

=============================================
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_sec_elution_csv(path_to_csv, absorbance):
    """ Read SEC elution .csv file and returns time and absorbance values."""
    
    df = pd.read_csv(path_to_csv, sep='\t', encoding='utf-16')
    time = df["Time"].values
    absorbance_values = df["{}".format(absorbance)].values
    
    return time, absorbance_values


def plot_sec_elution(time, absorbance_values, absorbance):
    """ Plot SEC elution profile."""
    
    fs = 22
    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111)
    
    ax.plot(time, absorbance_values, linewidth=5)
    ax.set_xlim(time.min(), time.max())
    
    ax.set_xlabel("Time (min)", fontsize=fs)
    ax.set_ylabel("Absorbance at {} (mAU)".format(absorbance), fontsize=fs)
    
    ax.tick_params(labelsize=fs)
    
    plt.tight_layout()
    plt.savefig("sec_elution_profile_at_{}.png".format(absorbance, dpi=400))
    
    return


if __name__ == "__main__":

	# Argument parser

	argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
	argparser.add_argument("-csv", type=str, help="Path to SEC elution .CSV file", required=True)
	argparser.add_argument("-abs", type=int, help="Absorbance wavelength", required=True)

	# Parse arguments

	args = argparser.parse_args()
	path_to_csv = args.csv
	absorbance_wavelength = args.abs

	# Plot SEC profile

	time, absorbance_values = read_sec_elution_csv(path_to_csv, absorbance_wavelength)
	plot_sec_elution(time, absorbance_values, absorbance_wavelength)

