#!/usr/bin/python
d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script provides a way to perform Guinier analysis
on the provided scattering scattering curve.

The scripts does the following:
(i) Upon loading a scattering curve file (.dat), one
defines the first and last point of Guinier region for analysis.
(ii) Then, a linear fit is done over the Guinier region and
associated values of R(g) and I(0) with their respective errors are
extracted. Additionally, values of q*R(g), q region, number of q points
and a linear quality of fit parameter are reported.

Before use:
- Install numpy, scipy, matplotlib.

To use the script provide:
- Experimental scattering curve file (.dat) together with 3 columns
of q, I(q) and sigma values.

The the units of scattering angle q must be in angstroms.


Marius Kausas					   2019 02 26

=============================================
"""


import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.style.use("bmh")


def linear(x, a, b):

	"""
	Function definition used for linear regression in scipy.optimize.curve_fit.

	"""
	return a * x + b


def guinier_lingress(x, y):

	"""
	Perform linear regression.

	Return a set of variables: slope, slope error, intercept, intercept error
	and last q point require for qRg limit determination.

	:param x: x variable data
	:param y: y variable data
	:return: slope, slope_error, intercept, intercept_error, last_point
	"""

	# Perfomr linear regression
	# popt - contains slope and intercept values
	# pcov - covariance matrix, where diagonals are variances of the slope and intercept

	popt, pcov = curve_fit(linear, x, y, method='lm')

	# Standard devations of linear regression variables

	perr = np.sqrt(np.diag(pcov))

	# Define variables

	slope, intercept = popt[0], popt[1]
	slope_error, intercept_error = perr[0], perr[1]

	# Define Correlation Coefficient (R) and Coefficient of Determination, (R^2)

	r = np.corrcoef(x, y)[0][1]
	r2 = r ** 2

	return slope, slope_error, intercept, intercept_error, r, r2


def guinier_properties(slope, slope_error, intercept, intercept_error, last_point):

	"""
	Calculate Guinier region properties (Rg, I0 and respective errors, and qRg limit)
	based on linear regression parameters.

	:param slope: Slope
	:param slope_error: Slope error
	:param intercept: Intercept
	:param intercept_error: Intercept error
	:param last_point: Last point required for qRg limit determination
	:return: I0, I0_error, Rg, Rg_error, qRg
	"""

	# Define I0 and error

	I0 = np.exp(intercept)
	I0_error = I0 * intercept_error

	# Define Rg and error

	Rg = np.sqrt(-3 * slope)
	Rg_error = 0.5 * Rg * ((slope_error) / (np.absolute(slope)))

	# Calculate qRg limit

	qRg = Rg * last_point

	return I0, I0_error, Rg, Rg_error, qRg


def guinier_plot(x, y, first_x_removed, first_y_removed,
				 last_x_removed, last_y_removed,
				 slope, intercept, I0, I0_error,
				 Rg, Rg_error, qRg, r2, output_name):

	"""
	Plot Guinier plot.

	Plot contains points used and excluded points from Guinier analysis;
	I0 and Rg wih their respective errors; qRg limit;
	linear fit and associated quality-of-fit metric.

	:param x: q region for Guinier analysis.
	:param y: I(q) region for Guinier analysis.
	:param first_x_removed: First x removed points.
	:param first_y_removed: First y removed points.
	:param last_x_removed: Last x removed points.
	:param last_y_removed: Last y removed points.
	:param slope: Slope for Guinier linear fit.
	:param intercept: Intercept for Guinier linear fit.
	:param I0: I0 value.
	:param I0_error: I0 error.
	:param Rg: Rg value.
	:param Rg_error: Rg error.
	:param qRg: q*Rg value of a Guinier analysis.
	:param r2: Qualiti-of-fit for linear fit.
	:param output_name: The name of the plot file.
	:return:
	"""

	# Plot Guinier

	fig = plt.figure(figsize=(10, 10))

	ax = fig.add_subplot(111)

	# Plot values used in Guinier analysis

	ax.plot(x, y, 'o', fillstyle="none", markersize=10, color="k")
	ax.plot(x, intercept + slope * x, 'r')

	# Plot first and last 30 excluded points for Guinier analysis

	ax.plot(first_x_removed, first_y_removed, 'o', fillstyle="none", markersize=10, color="b")
	ax.plot(last_x_removed, last_y_removed, 'o', fillstyle="none", markersize=10, color="b")

	ax.set_xlabel("$q^{2}$ ($\AA^{-2}$)", fontsize=25)
	ax.set_ylabel("$ln(I(0))$", fontsize=25)
	ax.tick_params(labelsize=20)

	ax.text(0.75, 0.95, s="$I(0)$ = {} $\pm$ {}".format(np.round(I0, 5), np.round(I0_error, 5)),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)
	ax.text(0.75, 0.85, s="$R_g$ = {} $\pm$ {}".format((np.round(Rg, 2)), np.round(Rg_error, 2)),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)
	ax.text(0.75, 0.75, s="$q*R_g$ limit = {}".format(np.round(qRg, 2)),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)
	ax.text(0.75, 0.65, s="$q$ region = {}-{}".format(np.round(x[0], 4), np.round(x[-1], 4)),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)
	ax.text(0.75, 0.55, s="Number of points used = {}".format(x.shape[0]),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)
	ax.text(0.75, 0.45, s="Quality-of-fit $R^2$ = {}".format(np.round(r2, 5)),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax.transAxes, fontsize=20)

	ax.set_title(output_name, fontsize=20)

	plt.savefig(output_name + ".png", dpi=300)
	plt.close()


def guinier_analysis(dat_file, first_point, last_point, output_name):

	"""
	Perfrom Guinier analysis on a given data set.

	:param dat_file: Scattering data set (three columns: q, I(0), sigma(I(0))).
	:param first_point: First q point to define beginning of Guinier region.
	:param last_point: Last q point to define end of Guinier region.
	:return: A plot containing Guinier region analysis.
	"""

	# Load scattering .dat file as a numpy array

	scattering = np.loadtxt(dat_file)

	# Define q^2 and ln(I(0)) required for Guinier analysis

	x = np.power(scattering[first_point - 1:last_point, :1], 2).T.squeeze()
	y = np.log(scattering[first_point - 1:last_point, 1:2]).T.squeeze()

	# First and last 30 removed points (for plotting)

	first_x_removed = np.power(scattering[:first_point - 1, :1], 2).T.squeeze()
	first_y_removed = np.log(scattering[:first_point - 1, 1:2]).T.squeeze()

	last_x_removed = np.power(scattering[last_point:last_point + 30, :1], 2).T.squeeze()
	last_y_removed = np.log(scattering[last_point:last_point + 30, 1:2]).T.squeeze()

	# Last point for sRg analysis

	last_point = scattering[last_point - 1:last_point, :1][0][0]

	# Linear regression

	slope, slope_error, intercept, intercept_error, r, r2 = guinier_lingress(x, y)

	# Guinier properties

	I0, I0_error, Rg, Rg_error, qRg = guinier_properties(slope, slope_error, intercept, intercept_error, last_point)

	# Plot Guinier

	guinier_plot(x, y,
				 first_x_removed, first_y_removed,
				 last_x_removed, last_y_removed,
				 slope, intercept,
				 I0, I0_error,
				 Rg, Rg_error,
				 qRg, r2, output_name)


if __name__ == "__main__":

	# Argument parser

	argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
	argparser.add_argument("-f", type=str, help="Path to the experimental file", required=True)
	argparser.add_argument("-first_point", type=int, help="First point for Guinier analysis", required=True)
	argparser.add_argument("-last_point", type=int, help="Last point for Guinier analysis", required=True)
	argparser.add_argument("-output_name", type=str, help="Output name for a file", required=True)

	# Parse arguments

	args = argparser.parse_args()
	path_to_file = args.f
	first_point = args.first_point
	last_point = args.last_point
	output_name = args.output_name

	# Do the Guinier analysis

	guinier_analysis(path_to_file, first_point, last_point, output_name)
