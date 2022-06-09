import os
import scipy 
import sys
from mpi4py import MPI
import numpy as np


from xspec.analysis import parse_dateset,SpectralAnalysis


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

complist = ["cutoffpl","xillver"]
path = os.getcwd()

foldername = sys.argv[1]
modelname = sys.argv[2]

os.chdir(f'{path}/spectra/{foldername}')
params = parse_dateset(f"{path}/scripts/spectralfiles.json",foldername)
#print(params)

Obs = SpectralAnalysis(path,modelname,*params)


#Obs.fit(n_live_points=1500)

if size == 1 :
    #Obs.simulate(n_spectra=1000,n_processes=10)
    Obs.plot()
   
    #Obs = SpectralAnalysis(path,modelname,*params)
    #Obs.lineSearch(minEnergy = 3,maxEnergy = 10,dE=0.01,n_Norm = 1000,minNorm = -0.2e-4,maxNorm=0.2e-4,redshift = 0.0053)
    # if "phenom" in modelname :
    #     if "NuSTAR" in params[3] and (params[1]=='p6' or params[1]=='p7') :
    #         Obs.get_flux(complist,minE=4,maxE=10,remove_spectra=[2,1]) #,
    #     elif "NuSTAR" in params[3] :
    #         Obs.get_flux(complist,minE=4,maxE=10,remove_spectra=[3,2,1])  #"cutoffpl"
    #     else :
    #         Obs.get_flux(complist,minE=4,maxE=10,remove_spectra=[2,1]) #,

