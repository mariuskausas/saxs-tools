d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script provides a way to analyse the effect of the contrast of the
hydration layer and number of harmonics used for CRYSOL calculations.

The scripts does the following:
(i) Calculate a set of scattering profiles for different combination of hydration shell contrast and harmonic orders.
(ii) Calculate RMSD between scattering profiles for each harmonic order.
(iii) Plot RMSD, scattering profiles, Kratky plots and summed residuals for given harmonic orders.

Before use:
- Install numpy, matplotlib.

The analysis scripts are based on the following reference:

Henriques, J. et al, On the Calculation of SAXS Profiles of Folded and
Intrinsically Disordered Proteins from Computer Simulations (2018), J Mol Bio, 430, 2521-2539.

Marius Kausas					   2019 10 28

=============================================
"""


import re
import os
import glob
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def system_command(command):

	"""
	Call a given command-line command.

	:param command: command-line command (str)
	:return:
	"""

	status = subprocess.call(command)

	return status


def run_crysol(pdb, h, q, dro, output):

	"""
	Run CRYSOL command.

	:param pdb: Path to the .pdb file (str)
	:param h: CRYSOL harmonic number (str)
	:param q: CRYSOL Maximum scattering vector in inverse angstroms (str)
	:param dro: CRYSOL contrast of the hydration layer (str)
	:param output: CRYSOL output name (str)
	:return:
	"""

	crysol_call = (["crysol"] + [pdb] +			# Input .pdb file
				["-lm"] + [h] +			# Number of harmonics
				["-fb"] +["17"] +		# Fibonnaci order
				["-sm"] + [q] +			# Maximum scattering vector
				["-ns"] + ["256"] +		# Number of points in theoretical curve
				["-dns"] + ["0.334"] +		# Solvent density
				["-dro"] + [dro] +		# Contrast of the hydration layer
				["-un"] + ["1"] +		# Angular units (inverse angstroms - 4*pi*sin(theta)/lambda
				["-err"] +			# Write experimental errors to .fit file
				["-cst"] +			# Constant substraction
				["-p"] + [output])		# Output name

	return system_command(crysol_call)


def run_crysol_harmonics(pdb, max_q, shell_contrasts, max_harmonic):

	"""
	Produce CRYSOL scattering based on given hydration shell contrasts and maximum harmonic order value.

	Produce a folder for each hydration shell contrast, containing scattering profiles
	for a different harmonic order value.

	:param pdb: Path to .pdb file (str)
	:param max_q: Maximum scattering vector in inverse angstroms (float)
	:param shell_contrasts: A list of hydration shell contrasts (floats)
	:param max_harmonic: Maximum number of harmonics.
	:return:
	"""

	# Calculate multiple CRYSOL-harmonic specific curves for each hydration shell contrast
	for shell_contrast in shell_contrasts:
		# Set hydration shell output folder
		dro = str(shell_contrast)
		dro_dest = "dro_" + dro.replace(".", "")
		os.mkdir(dro_dest)

		# Generate list of harmonics
		harmonics = list(np.arange(0, max_harmonic) + 1)

		for harmonic in harmonics:
			# Prepare CRYSOL parameters
			h = str(harmonic)
			max_q = str(max_q)
			run_crysol(pdb, h, max_q, dro, h)

			# Move CRYSOL output
			crysol_output = glob.glob(h + ".*")
			for file in crysol_output:
				shutil.move(file, dro_dest)

	return


def load_scatter(path_to_file):

	"""
	Load CRYSOL .fit output file.

	:param path_to_file: Path to fit file.
	:return:
	"""

	fit = np.loadtxt(path_to_file, skiprows=1)

	return fit


def analyse_single_harmonics(path_to_files, reference_harmonic):

	"""
	Perform RMSD analysis for given set of harmonics.

	:param path_to_files: Path to a directory with .fit files with different harmonics
	:param reference_harmonic: Reference harmonic .fit file (highest harmonic)
	:return:
	"""

	# Prepare reference harmonic scatter
	reference = load_scatter(reference_harmonic)[:, 1:2]
	scatters = glob.glob(path_to_files + "*.int")

	# Calculate RMSD of all harmonic fits to a reference fit
	rmsd_input = np.zeros((reference.shape[0], len(scatters)))

	for scatter in scatters:
		idx = int(re.findall(r"\d+", os.path.basename(scatter))[0])
		rmsd_input[:, idx - 1:idx] = load_scatter(scatter)[:, 1:2]

	differece = rmsd_input - reference
	rmsd = np.sqrt(np.sum(np.power(differece, 2), axis=0) / reference.shape[0])

	return rmsd


def visualize_harmonic_rmsd(rmsds):

	"""
	Plot RMSD for each harmonic order parameter.
	"""

	fs = 22
	fig = plt.figure(figsize=[6, 4])
	ax = fig.add_subplot(111)

	lables = ["$\\delta\\rho$ = 0 $e/\\AA^{3}$",
			"$\\delta\\rho$ = 0.01 $e/\\AA^{3}$",
			 "$\\delta\\rho$ = 0.02 $e/\\AA^{3}$",
			 "$\\delta\\rho$ = 0.03 $e/\\AA^{3}$"]

	for indx, rmsd in enumerate(rmsds):
		ax.plot(rmsd, linewidth=2, label=lables[indx])

	ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

	ax.set_xlim(10, 30)
	ax.set_ylim(0, 100)

	ax.vlines(25, 0, 1000, linestyle='--')
	ax.set_xlabel("Maximum order of harmonics", fontsize=fs)
	ax.set_ylabel("RMSD", fontsize=fs)

	plt.grid(linestyle='--')

	ax.legend(ncol=1, fontsize=10)

	plt.tight_layout()
	plt.savefig("harmonic_order_rmsd.png", dpi=400)

	return


def visualize_harmonic_scattering(path_to_files):

	"""
	Plot scattering profiles for each harmonic order parameter.

	:param path_to_files: Path to harmonic fits
	:return:
	"""

	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	scatters = glob.glob(path_to_files + "*.int")
	scatters.sort()

	q = load_scatter(scatters[0])[:, :1]

	lower = 15
	upper = 31
	i = np.arange(lower, upper)
	cmap = plt.cm.tab20(np.arange(upper - lower))

	for scatter in scatters:
		scatter_idx = int(re.findall(r"\d+", os.path.basename(scatter))[0])
		if scatter_idx in i:
			intensity = load_scatter(scatter)[:, 1:2]
			ax.plot(q, intensity, label=scatter_idx, color=cmap[scatter_idx - lower])

	ax.semilogy()

	ax.set_xlabel("q [$\\AA^{-1}$]", fontsize=fs)
	ax.set_ylabel("$I(q)$", fontsize=fs)

	plt.grid(linestyle='--')

	ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5), title="Maximum order harmonics")

	plt.title("Maximum order harmonic scattering for $\\delta\\rho$ = 0 $e/\\AA^{3}$")

	plt.tight_layout()
	plt.savefig("harmonic_order_scattering.png", dpi=400)

	return


def visualize_harmonic_kratky(path_to_files):

	"""
	Plot Kratky plot for each harmonic order parameter.

	:param path_to_files: Path to harmonic fits
	:return:
	"""

	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	scatters = glob.glob(path_to_files + "*.int")
	scatters.sort()

	q = load_scatter(scatters[0])[:, :1]

	lower = 15
	upper = 31
	i = np.arange(lower, upper)
	cmap = plt.cm.tab20(np.arange(upper - lower))

	for scatter in scatters:
		scatter_idx = int(re.findall(r"\d+", os.path.basename(scatter))[0])
		if scatter_idx in i:
			intensity = load_scatter(scatter)[:, 1:2]
			ax.plot(q, q ** 2 * intensity, label=scatter_idx, color=cmap[scatter_idx - lower])

	ax.set_xlabel("q [$\\AA^{-1}$]", fontsize=fs)
	ax.set_ylabel("$q^2I(q)$", fontsize=fs)

	plt.grid(linestyle='--')

	ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5), title="Maximum order harmonics")

	plt.title("Maximum order harmonic Kratky for $\\delta\\rho$ = 0 $e/\\AA^{3}$")

	plt.tight_layout()
	plt.savefig("harmonic_order_kratky.png", dpi=400)

	return


def visualize_harmonic_residuals(path_to_files, reference_harmonic, lower_harmonic, upper_harmonic):

	"""
	Plot RMSD plot of each harmonic order parameter.

	:param path_to_files: Path to harmonic fits
	:param reference_harmonic: Path to reference harmonic (highest)
	:param lower_harmonic: Lower bound for harmonic number to plot (int)
	:param upper_harmonic: Upper bound for harmonic number to plot (int)
	:return:
	"""

	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	scatters = glob.glob(path_to_files + "*.int")
	scatters.sort()

	reference = load_scatter(reference_harmonic)[:, 1:2]

	q = load_scatter(scatters[0])[:, :1]

	i = np.arange(lower_harmonic, upper_harmonic)

	cmap = plt.cm.tab20(np.arange(upper_harmonic - lower_harmonic))

	for scatter in scatters:
		scatter_idx = int(re.findall(r"\d+", os.path.basename(scatter))[0])
		if scatter_idx in i:
			intensity = load_scatter(scatter)[:, 1:2]
			ax.plot(q, (reference - intensity), label=scatter_idx, color=cmap[scatter_idx - lower_harmonic])
			ax.set_ylim(0, 1000)

	ax.set_xlabel("q [$\\AA^{-1}$]", fontsize=fs)
	ax.set_ylabel("$\\Delta$ $I_{50-h}(q)$", fontsize=fs - 5)

	plt.grid(linestyle='--')

	ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5), title="Maximum order harmonics")

	plt.title("Maximum order harmonic residuals for $\\delta\\rho$ = 0 $e/\\AA^{3}$")

	plt.tight_layout()
	plt.savefig("harmonic_order_residuals.png", dpi=400)

	return


def visualize_harmonic_summed_residuals(path_to_files, reference_harmonic, lower_harmonic, upper_harmonic):

	"""
	Plot summed up residuals between experimental and theoretical fits for each harmonic order.

	:param path_to_files: Path to harmonic fits
	:param reference_harmonic: Path to reference harmonic (highest)
	:param lower_harmonic: Lower bound for harmonic number to plot (int)
	:param upper_harmonic: Upper bound for harmonic number to plot (int)
	:return:
	"""

	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	scatters = glob.glob(path_to_files + "*.int")
	scatters.sort()

	reference = load_scatter(reference_harmonic)[:, 1:2]

	i = np.arange(lower_harmonic, upper_harmonic)

	summed_residuals = []

	count = 0

	for scatter in scatters:
		scatter_idx = int(re.findall(r"\d+", os.path.basename(scatter))[0])
		if scatter_idx in i:

			intensity = load_scatter(scatter)[:, 1:2]
			summed_residuals.append(np.sum(reference - intensity))

			if summed_residuals[count] == 0:
				ax.scatter(scatter_idx, summed_residuals[count],
						marker="o",
						s=50,
						label=scatter_idx,
						color="tab:grey",
						edgecolors="tab:red",
						linewidths=2)
			else:
				ax.scatter(scatter_idx, summed_residuals[count],
						marker="o",
						s=50,
						label=scatter_idx,
						color="tab:grey")

			count += 1

	ax.set_xlabel("Maximum harmonic order", fontsize=fs)
	ax.set_ylabel("$\\sum \\Delta$ $I_{50-h}(q)$", fontsize=fs - 5)

	ax.set_xticks(i)

	plt.grid(linestyle='--')

	ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5), title="Maximum order harmonics")

	plt.title("Summed residuals for maximum order harmonic $\\delta\\rho$ = 0 $e/\\AA^{3}$")

	plt.tight_layout()
	plt.savefig("harmonic_order_summedresiduals.png", dpi=400)

	return
