#!/usr/bin/python

d = """
=============================================

Analysis of small-angle X-ray scattering data.

The script provides a way to investigate fit residuals between
an experimental curve and theoretically calculated one.

The scripts does the following:
(i) Plot experimental and theoretically scattering curves.
(ii) Plot normalised fit residuals.
(iii) Plot the distribution of normalised fit residuals.
(iv) Plot Normal-Probability plot for normalised fit
residuals. The Normal-Probability plot serves as a visual
test for distribution normality.

Before use:
- Install numpy, scipy, matplotlib, pandas and seaborn.

To use the script provide:
- A CRYSOL/EOM output file (.fit) containing 4 columns (q, I(q), sigma and fit).

The the units of scattering angle q must be in angstroms.


Marius Kausas					   2019 02 25

=============================================
"""

import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("bmh")


def load_fit(path_to_file):

	"""
	Load a CRYSOL/EOM (.fit) file containing 4 columns (q, I(q), sigma and fit).

	:param path_to_file: Path to the .fit file (str).
	:return: .fit file loaded as a pandas DataFrame.
	"""

	fit = pd.read_csv(path_to_file,
							delim_whitespace=True,
							skiprows=1,
							names=["q", "Iqexp", "sigmaexp", "fit"])
	return fit


def chi2red(exp, theor, error):

	"""
	Calculate reduced chi square value between two scattering curves.

	:param exp: Experimental scattering intensities over q values.
	:param theor: Theoretical scattering intensities over q values.
	:param error: Experimental errors.
	:return: A float number for reduced chi square value.
	"""

	chi_value = np.sum(np.power((exp - theor) / error, 2)) / (exp.size - 1)

	return np.sum(chi_value)


def normprob_test(residuals):

	"""
	Helper function to perform a Normal-Probability test.

	:param residuals: Input data from a certain distribution.
	:return: results of stats.probplot and boundary were to draw a fit line.
	"""

	normprob = stats.probplot(residuals)
	boundary = np.round(max(normprob[0][1])) + 1

	return normprob, boundary


def normprobplot(normprob, boundary):

	"""
	Helper function to plot a Normal-Probability plot.

	:param normprob: First output of normprob_test()
	:param boundary: Second output of normprob_test()
	:return:
	"""

	fit_line = np.arange(-boundary, boundary + 1)
	plt.scatter(normprob[0][1], normprob[0][0], zorder=1, color="tab:red", s=1)
	plt.plot(fit_line, fit_line, color="k", linestyle='--', zorder=2, linewidth=1)
	plt.xlim(-boundary, boundary)
	plt.ylim(-boundary, boundary)


def plot_residuals(q, exp, fit, sigma, maxq, dataname, fitname, output_name):

	"""
	Plot scattering fit, residuals, distribution of residuals and Normal-probability plot of residuals.

	:param q: q angles of the experimental scattering curve.
	:param exp: Experimental scattering intensities over q values.
	:param fit: Theoretical scattering intensities over q values.
	:param sigma: Experimental errors.
	:param maxq: A max q value to be used for plotting.
	:param dataname: Name given for the experimental data.
	:param fitname: Name given for the theoretical data or the ensemble fit.
	:param output_name: Output name to be used in plotting as a title and saving plot.
	:return:
	"""

	# Calculate a reduced chi square value

	chi_value = str(np.around(chi2red(exp, fit, sigma), decimals=2))

	# Plot a fit between experimental and theoretical scattering data

	ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=3, rowspan=3)
	ax1.plot(q, np.log10(exp), label=dataname, color="k", linewidth=1)
	ax1.plot(q, np.log10(fit), label=fitname, color="tab:red", linewidth=1)
	ax1.tick_params(labelsize=6)
	ax1.set_xticklabels([])
	ax1.set_xlim(0, maxq)
	ax1.set_ylabel('$log_{10}(I_{q})$')
	ax1.legend(loc="upper right", fontsize=6)
	ax1.text(0.75, 0.75, s=('$\chi^{2}=$' + chi_value),
			 horizontalalignment='center',
			 verticalalignment='center',
			 transform=ax1.transAxes)
	ax1.set_title("Theoretical fit", fontsize=8)

	# Plot fit residuals

	residuals = (exp - fit) / sigma

	ax2 = plt.subplot2grid((4, 5), (3, 0), colspan=3)
	ax2.axhline(y=0, xmin=0, xmax=1, ls='--', color="k", zorder=2, linewidth=1)
	ax2.axhline(y=3, xmin=0, xmax=1, ls='--', color="k", zorder=2, linewidth=1)
	ax2.axhline(y=-3, xmin=0, xmax=1, ls='--', color="k", zorder=2, linewidth=1)
	ax2.scatter(q, residuals, s=1, color="tab:red", zorder=3)
	ax2.tick_params(labelsize=6)
	ax2.set_xlim(0, maxq)
	ax2.set_xlabel('$q$')
	ax2.set_yticks([-3, 3])
	ax2.set_ylabel('$(I\Delta)/\sigma$')
	ax2.set_title("Residuals", fontsize=8)

	# Calculate a Normal-Probability values and boundary for plotting

	normprob, boundary = normprob_test(residuals)

	# Plot distribution of residuals

	ax3 = plt.subplot2grid((4, 5), (0, 3), colspan=2, rowspan=2)
	sns.distplot(residuals, color="tab:red")
	ax3.tick_params(labelsize=6)
	ax3.set_ylabel("$P$", fontsize=8)
	ax3.set_xlim(-boundary, boundary)
	ax3.set_title("$(I\Delta)/\sigma$ distribution", fontsize=8)

	# Plot a Normal-Probability plot

	ax4 = plt.subplot2grid((4, 5), (2, 3), colspan=2, rowspan=2)
	normprobplot(normprob, boundary)
	ax4.tick_params(labelsize=6)
	ax4.set_xlabel("Ordered values of $(I\Delta)/\sigma$", fontsize=8)
	ax4.set_ylabel("Standard normal quantiles", fontsize=6)
	ax4.set_title("Normal-Probability plot", fontsize=8)

	plt.subplots_adjust(hspace=0.75, wspace=0.75)
	plt.suptitle(output_name)
	plt.savefig(output_name + ".png", dpi=600)
	plt.show()


if __name__ == "__main__":

	# Argument parser

	argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
	argparser.add_argument("-f", type=str, help="Path to the fit file (.fit)", required=True)
	argparser.add_argument("-exp_name", type=str, help="Experimental data name", required=True)
	argparser.add_argument("-fit_name", type=str, help="Fit data name", required=True)
	argparser.add_argument("-maxq", type=float, default=0.5, help="Maximum value of q for plotting, default=0.5 (A)", required=False)
	argparser.add_argument("-output_name", type=str, help="Output file name", required=True)

	# Parse arguments

	args = argparser.parse_args()
	path_to_file = args.f
	exp_name = args.exp_name
	fit_name = args.fit_name
	maxq = args.maxq
	output_name = args.output_name

	# Load the .fit file

	fit = load_fit(path_to_file)

	# Plot the residuals

	plot_residuals(q=fit['q'],
				exp=fit['Iqexp'],
				fit=fit['fit'],
				sigma=fit['sigmaexp'],
				maxq=maxq,
				dataname=exp_name,
				fitname=fit_name,
				output_name=output_name)



