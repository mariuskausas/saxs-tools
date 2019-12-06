#!/usr/bin/python
d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script allows to a normalized Kratky plot for a given .dat file.

The scattering file .dat must have three columns (q, I(q) and experimental errors).
Currently, the experimental errors are not taken into account when plotting

Before use:

- Install numpy, matplotlib.

Marius Kausas					   2019 12 06

=============================================
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_dat(path_to_dat):
    """ Load scattering .dat file"""
    
    dat = np.loadtxt(path_to_dat)
    q = dat[:,0]
    iq = dat[:,1]
    sigma = dat[:,2]
    
    return q, iq, sigma


def normalized_kratky(q, iq, i0, rg):
    """ Normalized Kratky function."""
    return (q * rg) ** 2 * iq/i0 


def guinier_approximation(x):
    """ Guinier approximation for a spherical particle."""
    return x ** 2 * np.exp(-x ** 2/3)


def plot_normalized_kratky(q, iq, i0, rg):
    """ Plot normalized Kratky plot."""
    
    fs = 22
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    
    marker_style = dict(c='tab:blue', marker='o', s=25, alpha=0.4, edgecolors="k")
    tick_params = dict(labelsize=fs, length=10, width=1)
    
    ax.scatter(q * rg,
        normalized_kratky(q, iq, i0, rg), 
        zorder=2,
        **marker_style)

    ax.plot(np.linspace(0, 15, 100), guinier_approximation(np.linspace(0, 15, 100)),
        zorder=2,
        linewidth=5,
        linestyle="--",
        color="k")
    
    ax.set_xlim(0,15)
    ax.set_ylim(0, 3)
    
    ax.tick_params(**tick_params)
    
    plt.tight_layout()
    plt.savefig("normalized_kratky_plot.png", dpi=300)
    plt.close()
    
    return


if __name__ == "__main__":

    # Argument parser

    argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
    argparser.add_argument("-dat", type=str, help="Path to scattering .dat file", required=True)
    argparser.add_argument("-i0", type=float, help="I(0) intensity at origin", required=True)	
    argparser.add_argument("-rg", type=float, help="Radius of gyration", required=True)

    # Parse arguments

    args = argparser.parse_args()
    path_to_dat = args.dat
    i0 = args.i0
    rg = args.rg

    # Plot normalized Kratky

    q, iq, sigma = load_dat(path_to_dat)
    plot_normalized_kratky(q, iq, i0, rg)

