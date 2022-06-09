import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import xspec
import re

plt.style.use("/home/mehdy/Nextcloud/Codes/Tools/2021.mplstyle")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def ml_spectra_plots(filename, expression, figsize=(10, 6.5), modelnumber=1, x_min=0, x_max=0, sigma_res=3, redshift=0,customlabels=None,colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]):
    """Function to plot spectra using PyXspec with the same synthax as XSPEC.

    Parameters
    ----------
    filename : str
        Name of the plot.
    expression : str
        Expression in the XSPEC synthax for the plot. Currently the supported values are :
        (eeufspec,eufspec,ufspec,ldata,data). If the deltastatistic needs to be plotted just add " de". 
        See examples below.
    figsize : tuple 
        Size of the figure in inches. Default value is (14,9).
    modelnumber : int 
        Group number of the model. Default value is 1.
    x_min : float
        Lower value for the Energy axes. Default is 0 means that the function will estimate this
        value according to the data.
    x_max : float
        Upper value for the Energy axes. Default is 0 means that the function will estimate this
        value according to the data.
    sigma_res : float
        Number of sigmas for the residuals plot. By default it is a 3-sigma residuals.
    redshift : float
        Redshift of the observation if the spectra needs to be plotted in the rest frame. Default value is 0
        so in the observed frame.

    Examples
    --------
    ml_plot_spectra("phenom","data de",(9,7))
    ml_plot_spectra("phenom","eeufspec de")
    ml_plot_spectra("phenom","ufspec de")
    ml_plot_spectra("phenom","eufspec")

    """
    old_dev = xspec.Plot.device
    xspec.Plot.device = '/null'
    xspec.Xset.chatter = 0
    n_spectra = xspec.AllData.nSpectra

    xspec.Plot.xAxis = 'keV'
    xspec.Plot.redshift = redshift
    xspec.Plot.area = True

    if redshift > 0:
        rest_en = 'Rest ~'
    else:
        rest_en = ''

    if 'de' in expression:
        fig, axs = plt.subplots(2, sharex=True, sharey=False, clear=True, gridspec_kw={
                                'hspace': 0, 'height_ratios': [2*1.618, 1.618]}, figsize=figsize, tight_layout=True)
        upper_panel = axs[0]
        lower_panel = axs[1]
    else:
        fig, axs = plt.subplots(1, sharex=True, sharey=False, clear=True, gridspec_kw={
                                'hspace': 0}, figsize=figsize, tight_layout=True)
        upper_panel = axs
    fig.patch.set_facecolor('blue')
    fig.patch.set_alpha(0.)
    if 'ldata' in expression:
        xspec.Plot("ldata")
        flux_unit = "\mathrm{counts}~~ \mathrm{s}^{-1}\,\mathrm{keV}^{-1}\mathrm{cm}^{-2}"
    elif 'data' in expression:
        xspec.Plot("data")
        flux_unit = "\mathrm{counts}~~ \mathrm{s}^{-1}\,\mathrm{keV}^{-1}\mathrm{cm}^{-2}"
    elif 'eeufspec' in expression:
        xspec.Plot("eeufspec")
        flux_unit = "\mathrm{keV}^2~ (\mathrm{Photons}~ \mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1})"
    elif 'eufspec' in expression:
        xspec.Plot("eufspec")
        flux_unit = "\mathrm{keV}~ (\mathrm{Photons}~ \mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1})"
    elif 'ufspec' in expression:
        xspec.Plot("ufspec")
        flux_unit = "\mathrm{Photons}~ \mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}"

    if n_spectra>=4 : ncol = 2
    else : ncol = 1
    for i in range(1, n_spectra+1):
        if customlabels != None :
            instrument = customlabels[i-1]
        else :
            instrument = re.search('[0-9]_(.*?)_grouped',xspec.AllData(i).fileName).group(1)
        x = np.array(xspec.Plot.x(i))
        y = np.array(xspec.Plot.y(i))
        modelnumber = i
        xErrs = xspec.Plot.xErr(i)
        yErrs = xspec.Plot.yErr(i)
        folded = np.array(xspec.Plot.model(i))
        linewidth = 1.5
        if x_min != 0 or x_max != 0:
            upper_panel.set_xlim(x_min-1e-2, x_max+1e-2)
        else:
            upper_panel.set_xlim(np.min(xspec.AllData(modelnumber).energies)*(1+redshift),
                                 np.max(xspec.AllData(modelnumber).energies)*(1+redshift)+1e-2)

        upper_panel.set_xscale("log")
        upper_panel.set_yscale("log")
        upper_panel.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())

        energies = np.array([ 3, 4, 5, 6, 7, 8, 9, 10, 20,  30, 40])
        if i>1 :
            oldxticks = xticks
            xticks = energies[-1+np.intersect1d(np.where(energies > np.min(xspec.AllData(modelnumber).energies)*(
                1+redshift)), np.where(energies < np.max(xspec.AllData(modelnumber).energies)*(1+redshift)))+1]
            if not len(xticks)>len(oldxticks):
                xticks = oldxticks
        else : 
            xticks = energies[-1+np.intersect1d(np.where(energies > np.min(xspec.AllData(modelnumber).energies)*(
                1+redshift)), np.where(energies < np.max(xspec.AllData(modelnumber).energies)*(1+redshift)))+1]
        if (np.min(xspec.AllData(1).energies) < 4) and (np.max(xspec.AllData(modelnumber).energies) < 11) :
            energies = np.array([ 3, 4, 5, 6, 7, 8, 9, 10])
            xticks = energies
        elif (np.min(xspec.AllData(1).energies) < 4) and (np.max(xspec.AllData(modelnumber).energies) > 25) :
            energies = np.array([ 3, 4, 5, 6, 7, 8, 9, 10,  20,  30, 40])
            xticks = energies
            
        upper_panel.set_xticks(xticks, minor=True)
        upper_panel.set_xticklabels(xticks, minor=True)

        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        upper_panel.get_xaxis().set_major_formatter(formatter)

        upper_panel.errorbar(x, y, xerr=xErrs, yerr=yErrs,
                             fmt='none', elinewidth=linewidth, capsize=0, ecolor=colors[i-1], label=f"{instrument}")
        upper_panel.step(x, folded, color=colors[i-1], linewidth=1.5*linewidth)
        upper_panel.set_ylabel(r"$"+flux_unit+"$")
        if not 'de' in expression: upper_panel.set_xlabel(r'$\mathrm{'+rest_en+' Energy~ (keV)}$')

        upper_panel.tick_params(axis="both", which="both",
                                direction="in", length=5, top=True, right=True)
        upper_panel.tick_params(axis="both", which="major",
                                direction="in", length=10, top=True, right=True)
        upper_panel.legend(frameon=False,ncol=ncol)
        upper_panel.patch.set_facecolor('white')
        upper_panel.patch.set_alpha(1)
        
    if 'de' in expression:
        xspec.Plot("delchi")
        for i in range(1, n_spectra+1):
            x = xspec.Plot.x(i)
            y = np.array(xspec.Plot.y(i))
            xErrs = xspec.Plot.xErr(i)
            yErrs = xspec.Plot.yErr(i)
            lower_panel.errorbar(x, y, yerr=yErrs, xerr=xErrs, fmt='none',
                                 elinewidth=linewidth, capsize=0, markersize=2., ecolor=colors[i-1])
            lower_panel.axhline(0., color="k", alpha=0.25)
            lower_panel.set_ylabel(
                r"$\displaystyle\frac{\mathrm{(data-model)}}{\mathrm{error}}$", labelpad=-10)
            lower_panel.set_ylim(-np.std(y)*sigma_res, np.std(y)*sigma_res)
            #lower_panel.set_ylim(np.min(y)-np.min(yErrs)
             #                    * 1.5, np.max(y)+np.max(yErrs)*1.5)
            lower_panel.tick_params(axis="both", which="both",
                                    direction="in", length=5, top=True, right=True)
            lower_panel.tick_params(
                axis="both", which="major", direction="in", length=10, top=True, right=True)
            lower_panel.set_xlabel(r'$\mathrm{'+rest_en+' Energy~ (keV)}$')
            lower_panel.patch.set_facecolor('white')
            lower_panel.patch.set_alpha(1)
    fig.align_labels()
    fig.savefig(f"{filename}_{expression}.pdf")

    xspec.Plot.device = old_dev
