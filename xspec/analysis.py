import xspec
import bxa.xspec as bxa
import importlib.util
from mpi4py import MPI
import os,json,re
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

plt.style.use("https://github.com/mlefkir/beauxgraphs/raw/main/beautifulgraphs.mplstyle")

import statistics
import spectra
from .addons import *
from .qq import *



class SpectralAnalysis :
    """
    This class is made to perform spectral analysis on a given set of spectra with a model :
    
    The directory tree is as follows :
    
    path/models/modelfile.py
    
    path/spectra/date_obsidA_observatory_portion
    path/spectra/date_obsidB_observatory_p1
    path/spectra/date_obsidB_observatory_p2
       
    
    
    Attributes
    ----------
    
    obsid : str or list of str 
        Observation identifier(s) of the set of spectra analyzed
    observatory : str or list of str 
        Observatories of the set of spectra 
    energy_bands : str or list of str 
        Energy bands of the spectra
    energy_ignore : str or list of str 
        Energies bands ignored in the analysis 
    n_spectra : int or list of int
        Number of spectra for each observatory
    instruments : list of str
        Name of the instruments for each spectra
    slices : list of str
        List of the slices in the observation. Should be ["all"] if the observation is not sliced
    optimal_bining : bool
        Use optimal binned spectra, default is False
    outputfiles_basename : str
        Name of the folder in the observation directory where the output files will be stored
    path : str
        Path to the observations directory
    portion : str
        Identifier for the portion of the observation. 
        "all" if the observation is not sliced and "p{int}" if it is sliced with {int} as the number of the slice
    date : str
        Date of the observation in the format YYYY-MM-DD
    priors : list of prior objects
        List of the priors for the model parameters 
    modelname : str
        Name of the model
    cstat_bf : list of float
        Best fit cstat value for each instrument
    dof_bf : list of int
        Best fit degree of freedom value for each instrument
    simultaneous : bool
        True if the observations is performed on several observations at the same time
    nb_free : int
        Number of free parameters in the model
    free_parameters : Dict
        List of the free parameters in the model
    
    
    Methods
    -------
    
    load_data(self, simulation=False, prefix="",suffix="",optimal_binning=False)
        Load spectra in PyXspec and ignore energy bands and bad channels
        
    load_model(self,modelfile)
        Load the model in PyXspec and set the priors
    
    fit(self,method="ultranest", **kwargs)
        Perform the fit with the selected method
        
    simulate(self,n_spectra=1000,n_processes=10)
        Simulate the model with the selected number of spectra
        
    generate_simu_spectra(self,nb_samples,prefix,**kwargs)
        Generate simulated spectra
        
    batch_simulations(self,prefix)
    
    get_stat_from_simu(self, prefix,suffix,optimal_binning=False,**kwargs)
    
    plot_simulations(self)

    get_free_pars(self) 
        Get the free parameters of the model 
        
    lineSearch(self,minEnergy,maxEnergy,dE,n_Norm,minNorm,maxNorm,redshift) 
        Perform a blind line search in the spectra using the loaded model
        
    get_flux(self,components_list,minE,maxE,remove_spectra=None)
        Get the flux of the selected components in the selected energy band
        
    plot(self,plot_type="ldata del") 
        Plot the spectra, model, residuals and QQ plots
    """
        
    def __init__(self,path,modelfile,obsid,portion,date,observatory,n_spectra,slices,energy_bands,energy_ignore,instruments,groupscale_constant,prefix="",optimal_binning = False) :
        """
        Initialize the class
        
        
        The constructor loads the data and the model and set the output files basename.
        
        Parameters
        ----------
        
        path : str
            Path to the observations directory
        modelfile : str
            Name of the model file
        obsid : str or list of str
            Observation identifier(s) of the set of spectra analyzed
        portion : str
            Identifier for the portion of the observation.
            "all" if the observation is not sliced and "p{int}" if it is sliced with {int} as the number of the slice
        date : str
            Date of the observation in the format YYYY-MM-DD
        observatory : str or list of str
            Observatory(ies) of the set of spectra
        n_spectra : int or list of int
            Number of spectra for each observatory
        slices : list of str
            List of the slices in the observation. Should be ["all"] if the observation is not sliced
        energy_bands : str or list of str
            Energy bands of the spectra
        energy_ignore : str or list of str
            Energies bands ignored in the analysis 
        instruments : list of str
            Name of the instruments for each spectra
        groupscale_constant : list of int
            Number of grouped channel for each observatory for the posterior predictive checks
        prefix : str
            Prefix for the output files
        optimal_binning : bool
            Use optimal binned spectra, default is False
            
        """
        # --- xspec initialisation ---
        xspec.Plot.device = "/null"
        xspec.Fit.statMethod = " cstat"
        xspec.Xset.chatter = 0

        # --- initialize the object ---
        self.simultaneous = (type(obsid) is list)
        if not self.simultaneous : 
            self.obsid = [obsid]
            self.observatory = [observatory]
            self.energy_bands = [energy_bands]
            self.energy_ignore = [energy_ignore]
        else : 
            assert len(observatory)==len(n_spectra), "Not the number of obsid and observatory"
            self.obsid = obsid
            assert len(observatory)==len(n_spectra), "Not the same number of spectra and observatory"
            self.observatory = observatory
            assert len(energy_bands)==len(energy_ignore), "Not the same number of ignored bands and energy bands"
            self.energy_bands = energy_bands
            self.energy_ignore = energy_ignore

            
        self.date = date
        self.slices = slices
        self.instruments = instruments
        self.groupscale_constant = groupscale_constant
        self.path = path
        self.n_spectra = n_spectra
        self.optimal_bining = optimal_binning      
        self.portion = portion
        self.load_data()
        self.load_model(modelfile)
        self.outputfiles_basename = f"{self.date}_{self.modelname}_{self.portion}"

    def load_data(self, simulation=False, prefix="",suffix="",optimal_binning=False):
        """
        Load spectra in PyXspec and ignore energy bands and bad channels
        
        The assumed name of the grouped spectra is :
        [prefix]obsid_instrument_grouped[optstr]_spectrum_energyband_portion.fits
        Note: The spectra are loaded in different data groups.
        
        
        Parameters
        ----------
        
        simulation : bool
            True if the data is simulated, default is False
        prefix : str
            Prefix for the output files, default is ""
        suffix : str
            Suffix for the output files, default is ""
        optimal_binning : bool
            Use optimal binned spectra, default is False       
        """
        
        xspec.AllData.clear()
        xspec.AllModels.clear()
        
        if optimal_binning: optstr = "_optim"
        else: optstr = ""
        data_String = ""
        igno_String = ""
        obs_counter = 0 # counter for the observatory parameters such as obsid, energybands, number of spectra with this observatory
        curr_counter = 1
        for counter,instr in enumerate(self.instruments,1) :
            
            if not simulation : filename = f"{prefix}{self.obsid[obs_counter]}_{instr}_grouped{optstr}_spectrum_{self.energy_bands[obs_counter]}_{self.portion}.fits "
            else : filename = f"{prefix}fakerebin_{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}_{suffix}.fak "
            
            igno_String += f"{counter}:**-{self.energy_ignore[obs_counter]}-** "
            data_String += f"{counter}:{counter} {filename} "
            if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
            curr_counter += 1
        xspec.AllData(data_String)
        xspec.AllData.ignore("bad")
        xspec.AllData.ignore(igno_String)
        
    def load_model(self,modelfile):
        """
        Load the model in XSPEC according to a Python model file, also load the priors and the modelname
        
        The model file is a Python file with the following structure :
        '''
        def xspec_model(**kwargs):
            modelname = "modelname"
            model = xspec.Model(mymodel)
            model.setPars() 
            priors = [prior1,prior2,...]
            return priors, modelname
        '''
               
        Parameters
        ----------
        modelfile : str
            Name of the model file
        """
        
        if os.path.isfile(f"{self.path}/models/{modelfile}.py"):
            spec = importlib.util.spec_from_file_location("module.name", f"{self.path}/models/{modelfile}.py")
        else :
            
            raise ValueError(f"Model file {modelfile}.py do not exist!")

        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        self.priors, self.modelname = model.xspec_model(obsid=self.obsid,portion=self.portion)
 
    def fit(self,method="ultranest", **kwargs):
        """
        Fit model to data with selected method.
        
        
        Parameters
        ----------
        method : str
            Method to fit the model to the data, default is "ultranest"
        **kwargs : dict
            Additional arguments for the fit method (e.g. speed, resume, n_live_points, etc.)
        
        """
        
        resume = kwargs.get('resume',True)
        n_live_points = kwargs.get('n_live_points',400)
        speed = kwargs.get('speed',"safe")
        
        solver = bxa.BXASolver(transformations=self.priors, outputfiles_basename=self.outputfiles_basename)
        
        if method == "ultranest" :
            results = solver.run(resume=resume, speed=speed, n_live_points=n_live_points)
            
        if MPI.COMM_WORLD.Get_rank() == 0 or MPI.COMM_WORLD.Get_size() == 1 :
            solver.set_best_fit()
            if os.path.isfile(f"{self.outputfiles_basename}.xcm") : os.remove(f"{self.outputfiles_basename}.xcm")
            if os.path.isfile(f"{self.outputfiles_basename}_model.xcm") : os.remove(f"{self.outputfiles_basename}_model.xcm")
            xspec.Xset.save(f"{self.outputfiles_basename}.xcm", info="a")
            xspec.Xset.save(f"{self.outputfiles_basename}_model.xcm", info="m")
            
    def plot(self,plot_type="ldata del"):
        """
        Plot spectra and qq plots
        
        Plot the spectra and residuals of the dataset with the best fit model.
        The QQ plots are also plotted using a modified version of the code of Johannes Buchner.
        The files are saved in the outputfiles_basename directory.
        
        
        Parameters
        ----------
        plot_type : str
            Type of plot to plot, default is ldata del
        """
        xspec.Xset.restore(f"{self.outputfiles_basename}.xcm")
        print(f"< INFO > : Plotting spectra")
        spectra.ml_spectra_plots(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}", plot_type)
        print(f"< INFO > : Plotting QQ plots")
        qq.qq(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}", markers=10, annotate=True)
        # save flux
        with open(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_flux.txt", 'w') as f:
            f.write(xspec.AllModels.calcFlux('2. 10.'))
    def batch_simulations(self,prefix):
        """
        Wrapper for analysing one batch of simulation.
        
        This function calls the get_stat_from_simu function and save the results in a text file.
    
        Parameters
        ----------
        prefix : str
            Prefix for the output files
        """
        with open(f"{prefix}simulations.txt","w") as file :
            for i in tqdm(range(1,self.n_samples_lim+1)) :  
                arr = self.get_stat_from_simu(prefix,f'{i}')
                file.write(f"{arr}".replace("[","").replace("]","").replace(", "," ")+"\n")
                file.flush()    
        
    def simulate(self,n_spectra=1000,n_processes=10):
        """
        Simulate n_samples spectra of the current model.
        
        
        This function calls the generate_simu_spectra to generate the simulated spectra and 
        then calls the get_stat_from_simu function to get the statistics in a parallelized way.
        It saves the results in a text file named 'outputfiles_basename_simulation.txt'
        
        Containing the following information :
        column 1 : cstat of the simulated spectrum in instrument 1
        column 2 : degree of freedom of the simulated spectrum in instrument 1
        and so on for all instruments
        
        This function also calls the plot_simu_spectra to plot the distribution of the statistics of the 
        simulated spectra against the best fit values.        
        
        Parameters
        ----------
        n_spectra : int
            Number of simulated spectra, default is 1000
        n_processes : int
            Number of processes to use for the simulations, default is 10       
        
        """
        
        assert os.path.isfile(f"{self.outputfiles_basename}.xcm"), "No XCM file found !"
        xspec.Xset.restore(f"{self.outputfiles_basename}.xcm")
        self.get_free_pars()

        cstat, bins  =  statistics.compute_cstat_with_background()
        cstat.append(sum(cstat))
        bins.append(sum(bins)-self.nb_free)
        self.cstat_bf, self.bins_bf = cstat, bins
        self.n_samples_lim = int(n_spectra/n_processes)
        prefix_list = [f"{x}_" for x in range(1,n_processes+1) ]

        if not os.path.isfile(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_simulation.txt") :


            processes = []
            print("< INFO > : Generate spectra ")
            for prefix in prefix_list :
                self.generate_simu_spectra(self.n_samples_lim,prefix)
                processes.append(mp.Process(target=self.batch_simulations, args=(prefix,)))
            print("< INFO > : Start evaluating cstat")
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            with open(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_simulation.txt","w") as simu_file :
                for fileidx in range(1,n_processes+1)  :
                    with open(f"{fileidx}_simulations.txt","r") as cfd_file :
                        simu_file.write(cfd_file.read())
                        simu_file.flush()
                    os.remove(f"{fileidx}_simulations.txt")
            # posterior predictive checks
            self.posterior_predictive_checks(prefix_list)
            self.remove_fake_files(prefix_list)
        self.plot_simulations()


    def remove_fake_files(self,prefix_list):
        obs_counter = 0 
        curr_counter = 1
        for counter,instr in enumerate(self.instruments) :
            resp_name = f"{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}"
            for prefix in prefix_list :
                for i in range(1,self.n_samples_lim+1): 
                    os.remove(f'{prefix}fake_{resp_name}_{i}.fak')
                    os.remove(f'{prefix}fakerebinconst_{resp_name}_{i}.fak')
                    os.remove(f"{prefix}fake_{resp_name}_{i}_bkg.fak")
                    os.remove(f"{prefix}fakerebin_{resp_name}_{i}.fak")
            filename = f"{self.obsid[obs_counter]}_{instr}_spectrum_src_{self.energy_bands[obs_counter]}_{self.portion}.fits"
            gpr_cst = f'grpcst_{filename}'
            os.remove(gpr_cst)
            if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
            curr_counter += 1
            # rebinning the original data

    def posterior_predictive_checks(self,prefix_list):
        """
        Compute the blabla for the current model.


        Parameters
        ----------
        
        """
        print("< INFO > : Posterior predictive checks from simulations")
        import matplotlib.ticker as ticker
        from matplotlib.ticker import FuncFormatter

        # rebinning the simulated spectra with a constant number of channels
        obs_counter = 0 
        curr_counter = 1
        for counter,instr in enumerate(self.instruments) :
            resp_name = f"{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}"
            for prefix in prefix_list :
                for i in range(1,self.n_samples_lim+1): 
                    if not os.path.isfile(f"{prefix}fakerebinconst_{resp_name}_{i}.fak") :
                        os.system(f"ftgrouppha infile='{prefix}fake_{resp_name}_{i}.fak' outfile='{prefix}fakerebinconst_{resp_name}_{i}.fak' grouptype=constant groupscale={self.groupscale_constant[obs_counter]}")
            # rebinning the original data
            filename = f"{self.obsid[obs_counter]}_{instr}_spectrum_src_{self.energy_bands[obs_counter]}_{self.portion}.fits"
            gpr_cst = f'grpcst_{filename}'
            if not os.path.isfile(f"{gpr_cst}") :
                rmf = filename.replace("_spectrum_src","").replace("fits","rmf")
                arf = filename.replace("_spectrum_src","").replace("fits","arf")
                bkg = filename.replace("src","bkg")
                os.system(f"ftgrouppha infile='{filename}' outfile='{gpr_cst}' grouptype=constant groupscale={self.groupscale_constant[obs_counter]}")
                os.system(f"grppha infile='{gpr_cst}' outfile='{gpr_cst}' clobber=yes comm='CHKEY BACKFILE {bkg}' tempc=exit")
                os.system(f"grppha infile='{gpr_cst}' outfile='{gpr_cst}' clobber=yes comm='CHKEY ANCRFILE {arf}' tempc=exit")
                os.system(f"grppha infile='{gpr_cst}' outfile='{gpr_cst}' clobber=yes comm='CHKEY RESPFILE {rmf}' tempc=exit")
            if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
            curr_counter += 1
        # plot the rebinned spectra
        if len(self.instruments) == 2 :
            figure,axs = plt.subplots(1, 2, figsize=(18, 4.7))
        elif len(self.instruments) == 3 :
            figure,axs = plt.subplots(2, 2, figsize=(18, 4.7*2))
        elif len(self.instruments) == 4 :
            figure,axs = plt.subplots(2, 2, figsize=(18, 4.7*2))
        elif len(self.instruments) == 5 :
            figure,axs = plt.subplots(3, 2, figsize=(18, 4.7*3))

        xspec.Plot.device = '/null'
        xspec.Xset.chatter = 0

        xspec.Plot.xAxis = 'keV'
        xspec.Plot.area = True

        modelnumber=1
        redshift=0
        flux_unit = "\mathrm{counts}~~ \mathrm{s}^{-1}\,\mathrm{keV}^{-1}\mathrm{cm}^{-2}"
        linewidth = 1.5
        obs_counter = 0
        curr_counter = 1

        for instr_idex,ax in enumerate(axs.flatten()):
            if instr_idex < len(self.instruments) :
                instr = self.instruments[instr_idex]
                resp_name = f"{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}"

                upper_panel = ax#fig.subplots(1, sharex=True, sharey=False,gridspec_kw={'hspace': 0})
                # fig.patch.set_facecolor('blue')
                # fig.patch.set_alpha(0.)
             
                for prefix in prefix_list:
                    for i in range(1,self.n_samples_lim+1): 
                        fakefile = f'{prefix}fakerebinconst_{resp_name}_{i}.fak'

                        xspec.AllData(fakefile)
                        igno_String = f"**-{self.energy_ignore[obs_counter]}-**"
                        xspec.AllData.ignore("bad")
                        xspec.AllData.ignore(igno_String)
                       
                        xspec.Plot("ldata")
                        x = np.array(xspec.Plot.x(1))
                        y = np.array(xspec.Plot.y(1))
                        xErrs = xspec.Plot.xErr(1)
                        upper_panel.errorbar(x, y, xerr=xErrs,
                                            fmt='none', elinewidth=linewidth, capsize=0, ecolor="k",alpha=0.02)

                filename = f"{self.obsid[obs_counter]}_{self.instruments[instr_idex]}_spectrum_src_{self.energy_bands[obs_counter]}_{self.portion}.fits"
                gpr_cst = f'grpcst_{filename}'
                xspec.AllData(gpr_cst)
                igno_String = f"**-{self.energy_ignore[obs_counter]}-**"
                xspec.AllData.ignore("bad")
                xspec.AllData.ignore(igno_String)

                xspec.Plot("ldata")
                x = np.array(xspec.Plot.x(1))
                y = np.array(xspec.Plot.y(1))
                xErrs = xspec.Plot.xErr(1)
                yErrs = xspec.Plot.yErr(1)

                upper_panel.errorbar(x, y, xerr=xErrs,yerr=yErrs,
                                    fmt='none', elinewidth=linewidth, capsize=0, ecolor="r")#, label=f"{instr}")
                upper_panel.set_ylim(0.7*np.min(y[y>0]),1.3*np.max(y[y>0])) 
                upper_panel.set_xlim(np.min(xspec.AllData(modelnumber).energies)*(1+redshift),np.max(xspec.AllData(modelnumber).energies)*(1+redshift)+1e-2)
                upper_panel.set_xscale("log")
                upper_panel.set_yscale("log")
                upper_panel.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())

                energies = np.array([ 3, 4, 5, 6, 7, 8, 9, 10, 20,  30, 40])
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

                upper_panel.set_ylabel(r"$"+flux_unit+"$")
                upper_panel.set_xlabel(r'$\mathrm{'+' Energy~ (keV)}$')

                upper_panel.tick_params(axis="both", which="both",
                                        direction="in", length=5, top=True, right=True)
                upper_panel.tick_params(axis="both", which="major",
                                        direction="in", length=10, top=True, right=True)
                #upper_panel.legend(frameon=False)
                upper_panel.set_title(instr)
                upper_panel.patch.set_facecolor('white')
                upper_panel.patch.set_alpha(1)


                if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
                curr_counter += 1
            else:
                ax.axis("off")
        figure.subplots_adjust(hspace=0.3, wspace=0.3)
        figure.savefig(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_posterior_simu.pdf",bbox_inches='tight')    

    def plot_simulations(self):
        """
        Plot the distribution of the statistics of the 
        simulated spectra against the best fit values of statistics for each instrument.      
        """
        simu_eval = np.loadtxt(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_simulation.txt")
        n_instr = len(self.instruments)
        
        #  plots
        if n_instr+1 == 3:
            fig, axes = plt.subplots(1, 3, figsize=(3*5, 6))
        elif n_instr+1==5 :
            fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(n_instr*3, 9))
            ax = axes[1:, -1][0]
            ax.remove()
        else:
            fig, axes = plt.subplots(
                2, int((n_instr+1)/2), figsize=(n_instr*3,8))

        i = 0
        l = 0
        labels = self.instruments.copy()
        labels.append("Total")
        from matplotlib.ticker import AutoMinorLocator


        for (ax, name) in zip(axes.flatten(), labels):
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())

    
            cstat = simu_eval[:, i]
            dof = simu_eval[:, i+1]
            dist, bins, obj = ax.hist(
                np.array(cstat)/np.array(dof), bins='auto')
            ax.vlines(self.cstat_bf[l]/self.bins_bf[l], ymin=0, ymax=np.max(dist),
                      linewidth=2, color='r', label='best fit')
            ax.set_xlabel("cstat/bins")
            ax.set_ylabel("Number of simulations")

            ax.title.set_text(name)
            i = i+2
            l = l+1
        ax.set_xlabel("cstat/dof")
        #ax.legend()
        fig.tight_layout()
        
        fig.savefig(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_simulations.pdf")
        plt.cla()
        
    def get_free_pars(self):
        """        
        Save all free parameters of each data group in a dictionary
        
        This function will find the number of free parameters of each data group
        and save them in a dictionary. In the following format : 
        
        self.free_parameters[data_group][par_index] = values
        
        """
        self.nb_free = 0
        self.free_parameters = {}
        self.frozen_parameters = {}
        for group in range(1,xspec.AllData.nGroups+1):
            if not ( f"{group}" in self.free_parameters.keys() ) :
                self.free_parameters[str(group)] = {} 
                self.frozen_parameters[str(group)] = {}
                
            for par_index in range(1,xspec.AllModels(group).nParameters+1):
                
                # if the parameter is not tied and is free 
                if (xspec.AllModels(group)(par_index).link=="") and (not xspec.AllModels(group)(par_index).frozen) :
                    self.free_parameters[str(group)][str(par_index)] = xspec.AllModels(group)(par_index).values
                    self.nb_free += 1
                    
                    # check if parameters in the next groups appart from the 1st one 
                    # are tied to this one, in that case we'll have to add them to the list
                    if group != 1 and group !=xspec.AllData.nGroups : 
                        for next_groups in range(group+1,xspec.AllData.nGroups+1):
                            for search_par_index in range(1,xspec.AllModels(next_groups).nParameters+1):
                                if xspec.AllModels(next_groups)(search_par_index).link == f"= p{par_index-1+xspec.AllModels(group).startParIndex}":
                                    self.free_parameters[str(next_groups)] = {}
                                    self.free_parameters[str(next_groups)][str(search_par_index)] = xspec.AllModels(group)(par_index).values
                elif xspec.AllModels(group)(par_index).frozen :
                    self.frozen_parameters[str(group)][str(par_index)] = xspec.AllModels(group)(par_index).values
                    
    def generate_simu_spectra(self,nb_samples,prefix,**kwargs):
        """
        Generate simulated spectra using the loaded model and group the spectra according to the options
        
        The simulated spectra are generated with Fakeit using the loaded model for all instruments and then grouped with ftgrouppha.
        
        
        Parameters
        ----------
        
        nb_samples : int
            Number of simulated spectra to generate
        prefix : str
            Prefix to add to the output files
        kwargs : dict
            Options to pass for the grouped spectra generation
            grouptype : str, default "min"
            groupscale : float, default 25
        """
        grouptype = kwargs.get('groupetype','min')
        groupscale = kwargs.get('groupscale',25)
        groupscale_constant = kwargs.get('groupscale_constant',50)
        obs_counter = 0 # counter for the observatory parameters such as obsid, energybands, number of spectra with this observatory
        curr_counter = 1
 
        for counter,instr in enumerate(self.instruments,1) :
            
            resp_name = f"{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}"
            fs = xspec.FakeitSettings(response=f"{resp_name}.rmf", arf=f"{resp_name}.arf",  background=f"{self.obsid[obs_counter]}_{instr}_spectrum_bkg_{self.energy_bands[obs_counter]}_{self.portion}.fits") 
            if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
            curr_counter += 1

            xspec.AllData.clear()

            # change values in the group 1 and save old parameters
            if  ( counter != 1 )  and ( len(self.free_parameters[f"{counter}"]) != 0 ):
                old_values = []
                for par_index in self.free_parameters[f"{counter}"].keys() :
                    old_values.append(xspec.AllModels(1)(int(par_index)).values)
                    xspec.AllModels(1)(int(par_index)).values = self.free_parameters[f"{counter}"][f"{par_index}"] 
            if  ( len(self.frozen_parameters[f"{counter}"]) != 0 ):
                for par_index in self.frozen_parameters[f"{counter}"].keys() :
                    xspec.AllModels(1)(int(par_index)).values = self.frozen_parameters[f"{counter}"][f"{par_index}"] 
                    xspec.AllModels(1)(int(par_index)).frozen = True        

            xspec.AllData.fakeit( nSpectra=nb_samples, settings=nb_samples*[fs], applyStats=True, filePrefix=f'{prefix}fake_')

            # restore the original values 
            if  ( counter != 1 )  and ( len(self.free_parameters[f"{counter}"]) != 0 ):
                assert len(old_values)==len(self.free_parameters[f"{counter}"].keys()), "Did not save enough values before the fakeit or the free parameters array was not correctly filled"
                for (old_value,par_index) in zip(old_values,self.free_parameters[f"{counter}"].keys()) :
                    xspec.AllModels(1)(int(par_index)).values = old_value
            # rebinning the spectra 
            for i in range(1,nb_samples+1):
                os.system(f"ftgrouppha infile='{prefix}fake_{resp_name}_{i}.fak' outfile='{prefix}fakerebin_{resp_name}_{i}.fak' grouptype={grouptype} groupscale={groupscale}")
                #os.remove(f"{prefix}fake_{resp_name}_{i}.fak")
    

    def get_stat_from_simu(self, prefix,suffix,optimal_binning=False,**kwargs) :
        """
        Get the list of cstat and degree of freedom for one set of spectra
        
        Parameters
        ----------
        prefix : str
            Prefix of the simulated spectra
        suffix : str
            Suffix of the simulated spectra
        optimal_binning : bool
            Use the optimal binning for the cstat calculation, default is False
        **kwargs : dict
            Additional options to pass to the cstat command
        
        """
        
        self.load_data(simulation=True,prefix=prefix,suffix=suffix)
        xspec.Xset.restore(f"{self.outputfiles_basename}_model.xcm")
        xspec.Fit.show()
        cstat, bins = statistics.compute_cstat_with_background()
        
        # obs_counter = 0
        # curr_counter = 1

        # for counter,instr in enumerate(self.instruments,1) :
        #     #os.remove(f"{prefix}fake_{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}_{suffix}_bkg.fak")
        #     #os.remove(f"{prefix}fakerebin_{self.obsid[obs_counter]}_{instr}_{self.energy_bands[obs_counter]}_{self.portion}_{suffix}.fak")
        #     if self.simultaneous and curr_counter == self.n_spectra[obs_counter] : obs_counter+=1;curr_counter=0
        #     curr_counter += 1
        nested = [[a, b] for a, b in zip(cstat, bins)]
        nested.append([sum(cstat),sum(bins)-self.nb_free])
        flattened = [element for sublist in nested for element in sublist]
        return  flattened
 
    def lineSearch(self,minEnergy,maxEnergy,dE,n_Norm,minNorm,maxNorm,redshift):
        """
        Perform a search for Gaussian Lines in the spectra.
        
        The Gaussian is moved in energy and a steppar is performed for the normalization.
        The delta statistics are stop at each step in energy. All the values are stored in
        a npy file and a map of the contours of the delta statistics is plotted.
        
        Parameters
        ----------
        minEnergy : float
            Minimum energy to search for lines, in keV
        maxEnergy : float
            Maximum energy to search for lines, in keV
        dE : float
            Energy step to search for lines, in keV
        n_Norm : int
            Number of points to use for the normalization of the Gaussian
        minNorm : float
            Minimum value for the normalization of the Gaussian
        maxNorm : float
            Maximum value for the normalization of the Gaussian
        redshift : float
            Redshift of the spectra
        """
        xspec.AllData.clear()
        xspec.AllModels.clear()
        print(f"< INFO > : Searching for lines in the spectrum")
        
        data_file = f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_LineSearch_data.npy"
        
        if not os.path.isfile(data_file):
            lineE_list = np.arange(minEnergy,maxEnergy+dE,dE)
            xspec.Xset.restore(f"{self.outputfiles_basename}.xcm")  
            cstat_bf = addons.old_addComp("zgauss")#addComp("zgauss")

            xspec.AllModels(1).zgauss.Redshift = (redshift , -1)
            xspec.AllModels(1).zgauss.Sigma = (0,-1)
            xspec.AllModels(1).zgauss.norm.values = (0,0.001,-1,-1,1,1)
            xspec.AllModels(1).zgauss.LineE.values = (minEnergy,0.01,minEnergy,minEnergy,maxEnergy+dE,maxEnergy+dE)

            cstat = []
            for energy in tqdm(lineE_list):
                xspec.AllModels(1).zgauss.LineE.values = (energy,-1)
                xspec.Fit.steppar(f"{xspec.AllModels(1).zgauss.norm.index} {maxNorm} {minNorm} {n_Norm}")
                cstat.append(xspec.Fit.stepparResults("statistic"))

            c = cstat_bf-np.array(cstat)
            c[c<0] = 0

            x,y = np.meshgrid(lineE_list,np.linspace(maxNorm,minNorm,n_Norm+1))
            np.save(data_file[:-4],np.array([x,y,c],dtype=object))
        else :
            x,y,c = np.load(data_file,allow_pickle=True)

        # plots
        fig,ax = plt.subplots(1,1,figsize=(12,6))
        contour = ax.contourf(x,y,c.T,cmap="Blues")
        plt.xlabel("Energy (keV)")
        plt.ylabel(r"Norm $(\mathrm{counts}~~ \mathrm{s}^{-1}\,\mathrm{keV}^{-1}\mathrm{cm}^{-2})$")
        cbar = plt.colorbar(contour)
        cbar.set_label(r'$\Delta$ '+xspec.Fit.statMethod)
        plt.hlines(0,minEnergy,maxEnergy,color="k",linestyle="--",alpha=0.5)
        ax.set_xlim(minEnergy,maxEnergy)
        ax.tick_params(axis="both", which="minor",bottom=True,left=True, top=True, right=True)
        ax.tick_params(axis="both", which="both",
                            direction="in", length=2.5, top=True, right=True)
        ax.tick_params(axis="both", which="major",
                            direction="in", length=7, top=True, right=True)        
        fig.tight_layout()
        fig.savefig(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_LineSearch_plot.pdf")
        xspec.Xset.restore(f"{self.outputfiles_basename}.xcm")  

    def get_flux(self,components_list,minE,maxE,remove_spectra=None):
        """
        Get the flux of the components in the spectra.
        
        Add a cflux model in front of the component and compute the flux in the given band.
        It is worth to note that the flux is computed for the band for all instruments meaning 
        that if the instruments have different bands the minE and maxE should coincide. Otherwise
        it is possible to remove spectra from the computation of the flux.
        
        Parameters
        ----------
        components_list : list of str
            List of the components to get the flux
        minE : float
            Minimum energy to compute the flux, in keV
        maxE : float
            Maximum energy to compute the flux, in keV
        remove_spectra : list of int
            List of the spectra to remove from the flux computation
        """
        
        if not os.path.isfile(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_flux_{minE}-{maxE}.txt") :
            print(f"< INFO > : Calculating fluxes of components")
            expression = xspec.AllModels(1).expression
            list_comp = get_components_list(expression)
            flux_comp = [list_comp.index(x)+1 for x in list_comp for comp in components_list if (comp in x)  ]
            flux_list = []
            for add_index in flux_comp :
                xspec.Xset.restore(f"{self.outputfiles_basename}.xcm")
                if remove_spectra != None :
                    for removed in remove_spectra :
                        xspec.AllData -= removed
                old_cstat = xspec.Fit.statistic
                xspec.Xset.chatter = 10
                xspec.AllModels.show()
                xspec.Fit.show()
                xspec.Xset.chatter = 0
                addons.addComp(add_index,"cflux")

                xspec.AllModels(1).cflux.Emin = (minE,-1)
                xspec.AllModels(1).cflux.Emax = (maxE,-1)
                xspec.Fit.perform()
                xspec.Xset.chatter = 10
                xspec.AllModels.show()
                xspec.Fit.show()
                xspec.Xset.chatter = 0
                new_cstat = xspec.Fit.statistic
                assert abs(new_cstat-old_cstat)<2, f"Cstat values differ too much !! new : {new_cstat} old : {old_cstat}"
                flux_list.append(10**xspec.AllModels(1).cflux.lg10Flux.values[0])
            np.savetxt(f"{self.outputfiles_basename}/{self.date[:4]}_{self.portion}_{self.modelname}_flux_{minE}-{maxE}.txt",np.array(flux_list),header= " ".join(components_list))
        else :
            print(f"< INFO > : Fluxes of components already computed")

def parse_dateset(json_file,foldername):
    """
    Retrieve the parameters needed for the spectral analysis object from a JSON file.
    
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing the parameters.
    foldername : str
        Path to the folder containing the observation files.
    
    Notes
    -----
    
    The json file must have the following parameters for a simultaneous observation and a single observation :
    
    [
    {"obsid":"obsid1,obsid2",
    "portion":"p5",
    "date":"YYYY-MM-DD",
    "observatory":"obs_1,obs_2",
    "n_spectra":"1,2",
    "slices":"7",
    "energy_bands":"0.5-10.0,3.0-40.0",
    "energy_ignore":"3.0 10.0,4.0 40.0",
    "instruments":"instr_A,instr_B,instr_C",
    "groupscale_constant":"value_A,value_B,value_C","}, 
    
    {"obsid":"obsid1",
    "portion":"all",
    "date":"YYYY-MM-DD",
    "observatory":"obs_1",
    "n_spectra":"3",
    "slices":"0",
    "energy_bands":"0.5-10.0",
    "energy_ignore":"3.0-10.0",
    "instruments":"instr_A,instr_B,instr_C",
    "groupscale_constant":""}
    ]
    """
    file = open(json_file)
    dataset = json.load(file)
    file.close()
    
    if '_p' in foldername:
        m = re.search('(.*\d)_(\w*)_(\d+)_(p[1-7])', foldername)
        year, observatory, obsid, portion = m.groups()
    else:
        m = re.search('(.*\d)_(\w*)_(\d+)', foldername)
        year, observatory, obsid = m.groups()
        portion = "all"
    
    for element in dataset:
        if obsid in element['obsid'] and portion == element['portion'] :
            obsid = list(element["obsid"].split(','))
            portion = element['portion']
            instruments = list(element["instruments"].split(','))
            date = element['date']
            observatory = list(element["observatory"].split(','))
            n_spectra =  [int(x) for x in list(element["n_spectra"].split(','))]
            slices = int(element["slices"])
            energy_bands =  list(element["energy_bands"].split(','))
            energy_ignore =  list(element["energy_ignore"].split(','))
            groupscale_constant = list(element["groupscale_constant"].split(','))
            break
    return [obsid,portion,date,observatory,n_spectra,slices,energy_bands,energy_ignore,instruments,groupscale_constant]
