#!/usr/bin/python3
d = """
=============================================

Generate averaged scattering profile for a given trajectory using CRYSOL (parallel implementation).

The script performs the following:
- Calculate theoretical profile for initial topology file and extract experimental scattering columns.
- Extract each trajectory frame as a .pdb file.
- Calculate theoretical profiles for each .pdb file and generate .fit file.
- Extract theoretical scattering from .fit file and add to a fit array.
- Average along fit array and join with previously extracted experimental scattering columns to produce final .fit file.
- Clean-up the CRYSOL .fit and .log files after each frame.

Before use:
- Install numpy, MDAnalysis and CRYSOL. Make sure crysol executable is in your path.

To use the script provide:
- Topology .pdb file of a starting trajectory frame.
- Trajectory .xtc file. 
- Experimental scattering curve file (.dat) together with 3 columns
of q, I(q) and sigma values.
- Number of processes or workers to run.

The the units of scattering angle q must be in angstroms.

Marius Kausas					   2019 07 12

=============================================
"""

import os
import argparse
import subprocess
import multiprocessing
import numpy as np
import MDAnalysis as mda


def system_command(command):
	""" Run a command line."""
	status = subprocess.call(command)
	return status


def run_crysol(pdb, exp_dat, index):
	""" Run CRYSOL."""
	crysol_call = (["crysol"] + [pdb] + [exp_dat] + ["-p"] + ["fit_" + str(index)])
	return system_command(crysol_call)


def read_crysol_fit(file):
	""" Read CRYSOL .fit output."""
	fit = np.loadtxt(file, skiprows=1)
	return fit


def get_exp_scattering(pdb, exp_dat):
	""" Extract angle, experimental scattering intensities and associated errors."""
	run_crysol(pdb, exp_dat, "initial")
	return read_crysol_fit("fit_initial.fit")[:, :3]


def get_fit_scattering(fit_file):
	""" Extract theoretical scattering from a .fit file."""
	return read_crysol_fit(fit_file)[:, 3:]


def clean_up(frame_pdb, frame_idx):
	""" Clean up after each CRYSOL calculation."""
	os.remove(frame_pdb)
	os.remove("fit_" + str(frame_idx) + ".fit")
	os.remove("fit_" + str(frame_idx) + ".log")
	return


def output_averaged_fit(exp_scatter, fit_array):
	""" Prepare output containing experimental data and averaged fit."""
	stacked_arr = np.hstack((exp_scatter, fit_array.mean(axis=0)))
	np.savetxt(fname="traj_averaged_scatter.fit", X=stacked_arr, fmt="%.6E", header=" ")
	return


if __name__ == "__main__":

	# Argument parser
	argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
	argparser.add_argument("-top", type=str, help="Path to topology file, e.g. .pdb", required=True)
	argparser.add_argument("-traj", type=str, help="Path to trajectory file, e.g. .xtc", required=True)
	argparser.add_argument("-exp_dat", type=str, help="Path to experimental scattering file .dat", required=True)
	argparser.add_argument("-workers", type=int, default=1, help="Number of processed to run", required=False)

	# Parse arguments
	args = argparser.parse_args()
	top = args.top
	traj = args.traj
	exp_dat = args.exp_dat
	workers = args.workers

	# Define MDAnalysis universe
	u = mda.Universe(top, traj)
	protein = u.select_atoms("all")

	# Extract experimental scattering for topology scattering fit file
	exp_scatter = get_exp_scattering(top, exp_dat)
	# Define array where to put all fit scattering
	fit_array = np.zeros((len(u.trajectory), exp_scatter.shape[0], 1))


	def sliced_traj_scatter(traj_slice):
		""" Return a fit array containing scattering profiles across all frames within a given trajectory slice."""
		# For each frame in trajectory, extract .pdb, calculate theoretical fit, add to fit array and clean-up .pdb/.fit
		for ts in traj_slice[:]:
			frame_idx = ts.frame
			u.trajectory[frame_idx]
			frame_pdb = str(frame_idx) + ".pdb"
			with mda.Writer(frame_pdb, protein) as W:
				W.write(protein)
			run_crysol(frame_pdb, exp_dat, frame_idx)
			fit_array[frame_idx] = get_fit_scattering("fit_" + str(frame_idx) + ".fit")
			clean_up(frame_pdb, frame_idx)
		return fit_array


	# Initialize multiprocessing
	pool = multiprocessing.Pool(workers)
	outputs = pool.map(sliced_traj_scatter, [u.trajectory[ts::workers] for ts in range(workers)])
	output_averaged_fit(exp_scatter, np.sum(outputs, axis=0))

