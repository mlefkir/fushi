#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From https://github.com/JohannesBuchner/BXA/blob/master/bxa/xspec/qq.py
# All rights to Johannes Buchner
# Modified by Mehdy Lefkir - Feb 2022

"""
Statistics and Plotting for quantile-quantile analysis for model discovery
"""

import numpy
import matplotlib.pyplot as plt
import os,re
from xspec import Plot,AllData

plt.style.use("/home/mehdy/Nextcloud/Codes/Tools/2021.mplstyle")

def KSstat(data, model):
	"""
	Kolmogorov-Smirnov statistic: maximum deviance between data and model
	"""
	modelc = model.cumsum() / model.sum()
	datac = data.cumsum() / data.sum()
	ks = numpy.abs(modelc - datac).max()
	return ks


def CvMstat(data, model):
	"""
	Cramér–von Mises statistic: Takes all deviances into account
	"""
	modelc = model.cumsum()
	datac = data.cumsum()
	maxmodelc = modelc.max()
	cvm = ((modelc / maxmodelc - datac / datac.max())
	       ** 2 * model / maxmodelc).mean()
	return cvm


def ADstat(data, model):
	"""
	Anderson-Darling statistic: Takes all deviances into account
	more weight on tails than CvM.
	"""
	modelc = model.cumsum()
	datac = data.cumsum()
	maxmodelc = modelc.max()
	valid = numpy.logical_and(modelc > 0, maxmodelc - modelc > 0)
	modelc = modelc[valid] / maxmodelc
	datac = datac[valid] / datac.max()
	model = model[valid] / maxmodelc
	assert (modelc > 0).all(), ['ADstat has zero cumulative denominator', modelc]
	assert (maxmodelc - modelc >
	        0).all(), ['ADstat has zero=1-1 cumulative denominator', maxmodelc - modelc]
	ad = ((modelc - datac)**2 / (modelc * (maxmodelc - modelc)) * model).sum()
	return ad


def qq_plot(bins, data, model, markers=[0.2, 1, 2, 5, 10], unit='', annotate=True):
	"""
	Create a quantile-quantile plot for model discovery (deviations in data from model).
	* bins: energies/channel
	* data: amount observed
	* model: amount predicted
	* markers: list of energies/channels (whichever the current plotting xaxis unit)
	* unit: unit to append to marker
	* annotate: add information to the plot
	"""

	assert numpy.isfinite(data).all(), data
	assert numpy.isfinite(model).all(), model
	datac = data.cumsum()
	modelc = model.cumsum()

	plt.plot(modelc, datac, color='red', drawstyle='steps', linewidth=3)

	for m in markers:
		mask = bins >= m
		# first true value
		i = mask.argmax()
		if mask[i]:
			plt.plot(modelc[i], datac[i], marker='+', color='k')
			if datac[i] < modelc[i]:
				textkwargs = dict(va='top', ha='left')
			else:
				textkwargs = dict(va='bottom', ha='right')
			plt.text(modelc[i], datac[i], '%s%s' % (m, unit), **textkwargs)

	u = max(datac[-1], modelc[-1])
	plt.plot([0, u], [0, u], ls='--', color='grey')
	plt.xlim(0, u)
	plt.ylim(0, u)
	plt.title('QQ-plot')
	plt.xlabel('integrated model')
	plt.ylabel('cumulative data counts')

	if annotate:
		plt.text(u/2, u/2, 'data excess', va='bottom',
		         ha='center', rotation=45, color='grey', size=8)
		plt.text(u/2, u/2, 'model excess', va='center',
		         ha='left', rotation=45, color='grey', size=8)
		stats = dict(
			ks=KSstat(data, model),
			cvm=CvMstat(data, model),
			ad=ADstat(data, model),
		)

		text = """K-S = %(ks).3f
C-vM = %(cvm).5f
A-D = %(ad)e""" % stats

		plt.text(u*0.98, u*0.02, text, va='bottom', ha='right', color='grey')


def xspecfilenamenormalise(path):
	"""
	Xspec gets a bit confused if there are "." in the filename
	So we replace . with _ in the filename.
	But if we replace it as part of the parent directory, we cannot write
	there, so this only alters the filename, not the entire path.
	"""
	if '/' in path:
		parts = path.split('/')
		parts = (parts[:-1] + [xspecfilenamenormalise(parts[-1])])
		return '/'.join(parts)
	return path.replace('.', '_')


def qq(prefix, markers=5, annotate=True):
	"""
	Create a quantile-quantile plot for model discovery (deviations in data from model).
	The current data and model is used, so call *set_best_fit(analyzer, transformations)*
	before, to get the qq plot at the best fit.
	* markers: list of energies/channels (whichever the current plotting xaxis unit)
	  or number of equally spaced markers between minimum+maximum.
	* annotate: add information to the plot
	"""

	olddevice = Plot.device
	Plot.xAxis = "keV"
	Plot.area = False
	Plot.device = '/null'
	tmpfilename = '%swdatatmp.qdp' # % xspecfilenamenormalise(prefix)
	if os.path.exists(tmpfilename):
		os.remove(tmpfilename)

	while len(Plot.commands) > 0:
		Plot.delCommand(1)
	Plot.addCommand('wdata "%s"' % tmpfilename.replace('.qdp', ''))

	Plot.background = True
	Plot("counts")
	
	# --- modifications start here ---
	fullcontent = numpy.genfromtxt(tmpfilename, skip_header=3)
	if os.path.exists(tmpfilename): os.remove(tmpfilename)
	while len(Plot.commands) > 0:
		Plot.delCommand(1)

	content = numpy.vsplit(
	    fullcontent, 1+numpy.unique(numpy.argwhere(numpy.isnan(fullcontent)).T[0]))  # +
	instr = 1
	for spectra_plot in content:
		fig = plt.figure(figsize=(7,7))
		instrument = re.search('[0-9]_(.*?)_grouped',AllData(instr).fileName).group(1)
		spectra_plot = spectra_plot[~numpy.isnan(spectra_plot)]
		
		if Plot.background:
			spectra_plot = numpy.reshape(spectra_plot, (int(spectra_plot.size/7), 7))
			bins, width, data, dataerror, background, backgrounderror, model = spectra_plot[:, 0:7].transpose()
		else:
			spectra_plot = numpy.reshape(spectra_plot, (int(spectra_plot.size/5), 5))
			bins, width, data, dataerror, model = spectra_plot[:, 0:5].transpose()
		if not hasattr(markers, '__len__'):
			nmarkers = int(markers)
			decimals = int(-numpy.log10(bins[-1] - bins[0] + 1e-4))
			markers = numpy.linspace(bins[0], bins[-1], nmarkers+2)[1:-1]
			markers = set(numpy.round(markers, decimals=decimals))

	    # make qq plot, with 1:1 line
		qq_plot(bins=bins, data=data * width * 2, model=model * width * 2, markers = markers, annotate = annotate, unit=Plot.xAxis)
		plt.savefig(f"{prefix}_{instrument}_qq_model_deviations.pdf", bbox_inches="tight")
		instr += 1
	Plot.device = olddevice
