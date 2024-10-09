import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import satlas as sat

magneticField = 4.48 # in mT
isotope = 'Xe129' # choose between H1, H2, Xe129, Xe131, Xe133, Xe129m, Xe131m, Xe133m

main_dir = '/Users/akanellako/Documents/NMR_data'
fileName = 'fft-TEST5_spherical_129Xe_2min_10pts_4000mV_391us_2.81A.txt'

def larmorFrequency(magneticField, isotope):
    if isotope == 'H1':
        magneticMoment = 2.79284734 # in nuclear magnetons
        nucSpin = 1/2
    elif isotope == 'H2':
        magneticMoment = 0.857438228 # in nuclear magnetons
        nucSpin = 1
    elif isotope == 'Xe129':
        magneticMoment = -0.7779763 # in nuclear magnetons
        nucSpin = 1/2
    elif isotope == 'Xe131':
        magneticMoment = 0.691862 # in nuclear magnetons
        nucSpin = 3/2
    elif isotope == 'Xe133':
        magneticMoment = 0.142 # in nuclear magnetons
        nucSpin = 3/2
    elif isotope == 'Xe129m':
        magneticMoment = -0.891223 # in nuclear magnetons
        nucSpin = 11/2
    elif isotope == 'Xe131m':
        magneticMoment = -0.994048 # in nuclear magnetons
        nucSpin = 11/2
    elif isotope == 'Xe133m':
        magneticMoment = 0.77 # in nuclear magnetons
        nucSpin = 11/2

    gFactor = magneticMoment/nucSpin

    gyromagneticRatio = (cts.physical_constants['nuclear magneton'][0]/cts.physical_constants['reduced Planck constant'][0]) * gFactor * (1e-6)/(2*math.pi)

    larmorFrequency = gyromagneticRatio * magneticField

    return abs(larmorFrequency) 

def NMRPeakModel(x, par):

    A = par[0]
    mu = par[1]
    FWHM = par[2]
    a0 = par[3]
    a1 = par[4]

    linear = (a1 * x) + a0

    lorentzian = A * (FWHM/2) / ( math.pi * ((x - mu)**2) + ((FWHM/2)**2) )


    return linear + lorentzian

def main(dataDir, resultDir, fileName):

    try: os.mkdir(resultDir)
    except FileExistsError: pass
    os.chdir(resultDir)

    dataFile = os.path.join(dataDir, fileName)
    resultDir = os.path.join(resultDir, fileName[:-4])

    try: os.mkdir(resultDir)
    except FileExistsError: pass
    os.chdir(resultDir)

    resultGraph = os.path.join(resultDir, '{}.png'.format(fileName[:-4]))

    inDF = pd.read_csv(dataFile, delimiter=' ', header=None)

    inDF.columns = ['frequency', 'intensity']
    inDF['intensityUnc'] = inDF['intensity']**0.5

    fLarmor = larmorFrequency(magneticField, isotope)
    cog = int(fLarmor)*1e3
    fLarmor = fLarmor*1e3

    fitDF = inDF[(inDF['frequency'] >= cog-5e3) & (inDF['frequency'] <= cog+5e3)]  

    # Initialisation of model parameters
    init_mu = fLarmor
    init_fwhm = 300.
    init_A = inDF.intensity.max()*init_fwhm*0.5
    init_a1 = -0.9
    init_a0 = 6000.
    init = [init_A, init_mu, init_fwhm, init_a0, init_a1]

    model = sat.MiscModel(NMRPeakModel, init, ['A', 'mu', 'FWHM', 'a0', 'a1'])
    model.set_boundaries({'a0': {'min': 0},
                            'a1': {'max': 0},
                            'A': {'min': 0}})

    success, message = sat.chisquare_fit(model, fitDF.frequency.to_numpy(), fitDF.intensity.to_numpy(), yerr=fitDF.intensityUnc.to_numpy())
    frame = model.get_result_frame()

    binPerHz = len(fitDF)/(fitDF.frequency.max()-fitDF.frequency.min())

    plt.figure(dpi = 200)
    plt.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensity)**0.5, label = "FFT", fmt='.', zorder=1)
    plt.plot(fitDF.frequency.to_numpy(), model(fitDF.frequency.to_numpy()), label = "Fit", linestyle = "--", zorder=2)
    plt.plot([fLarmor, fLarmor], [fitDF.intensity.min(), fitDF.intensity.max()])
    plt.xlabel("Frequence [Hz]")
    plt.ylabel("Intensity [a.u.]")
    plt.savefig(resultGraph)


#%% Main

# main_dir
# |
# |__result_path
#    |__run
# |
# |__data_dir
#    |__run

data_dir = os.path.join(main_dir, 'data')
result_dir = os.path.join(main_dir, 'result')

main(data_dir, result_dir, fileName)


