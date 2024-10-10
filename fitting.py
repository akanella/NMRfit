import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import satlas as sat
from uncertainties import ufloat

magneticField = 4.48 # in mT
isotope = 'Xe129' # choose between H1, H2, Xe129, Xe131, Xe133, Xe129m, Xe131m, Xe133m

main_dir = '/Users/akanellako/Documents/NMR_data'
fileName = 'fft-TEST5_spherical_129Xe_2min_10pts_4000mV_391us_2.81A.txt'

def larmorFrequency(magneticField, isotope):
    if isotope == 'H1':
        magneticMoment = uflaot(2.792847351, 0.000000009)  # in nuclear magnetons
        nucSpin = 1/2
    elif isotope == 'H2':
        magneticMoment = ufloat(0.857438231, 0.000000005) # in nuclear magnetons
        nucSpin = 1
    elif isotope == 'Xe129':
        magneticMoment = ufloat(-0.777961, 0.000016) # in nuclear magnetons
        nucSpin = 1/2
    elif isotope == 'Xe131':
        magneticMoment = ufloat(+0.691845, 0.000007) # in nuclear magnetons
        nucSpin = 3/2
    elif isotope == 'Xe133':
        magneticMoment = ufloat(+0.81335, 0.00007) # in nuclear magnetons
        nucSpin = 3/2
    elif isotope == 'Xe129m':
        magneticMoment = ufloat(-0.891170, 0.00001010) # in nuclear magnetons
        nucSpin = 11/2
    elif isotope == 'Xe131m':
        magneticMoment = ufloat(-0.993989, 0.000012) # in nuclear magnetons
        nucSpin = 11/2
    elif isotope == 'Xe133m':
        magneticMoment = ufloat(-1.08241, 0.00015) # in nuclear magnetons
        nucSpin = 11/2

    gFactor = magneticMoment/nucSpin

    gyromagneticRatio = (ufloat(cts.physical_constants['nuclear magneton'][0], cts.physical_constants['nuclear magneton'][2])/ufloat(cts.physical_constants['reduced Planck constant'][0], cts.physical_constants['reduced Planck constant'][2])) * gFactor * (1e-6)/(2*math.pi)

    larmorFrequency = gyromagneticRatio * ufloat(magneticField, 0.005)

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

    fitFile = os.path.join(resultDir, 'fit_{}.csv'.format(fileName[:-4]))
    resultFile = os.path.join(resultDir, '{}.csv'.format(fileName[:-4]))
    resultGraph = os.path.join(resultDir, '{}.png'.format(fileName[:-4]))

    inDF = pd.read_csv(dataFile, delimiter=' ', header=None)
    inDF.columns = ['frequency', 'intensity']

    fLarmor = larmorFrequency(magneticField, isotope)
    cog = int(fLarmor.nominal_value)*1e3
    fLarmor = fLarmor*1e3

    fitDF = inDF[(inDF['frequency'] >= cog-5e3) & (inDF['frequency'] <= cog+5e3)]
    fitDF['intensity'] = fitDF['intensity']/1e3
    aDF = fitDF[(fitDF['frequency'] <= cog-2e3)]
    bDF = fitDF[(fitDF['frequency'] >= cog+2e3)]
    cDF = pd.concat([aDF, bDF])
    std = cDF.intensity.std()
    fitDF['SNR'] = fitDF.intensity.abs()/std
    fitDF['intensityUnc'] = fitDF.intensity/fitDF.SNR
    fitDF.to_csv(fitFile, index = False)

    # Initialisation of model parameters
    init_mu = fLarmor.nominal_value
    init_fwhm = 300.
    init_A = inDF.intensity.max()*init_fwhm*0.5
    init_a1 = -0.9
    init_a0 = inDF.intensity.min()
    init = [init_A, init_mu, init_fwhm, init_a0, init_a1]

    model = sat.MiscModel(NMRPeakModel, init, ['A', 'mu', 'FWHM', 'a0', 'a1'])
    model.set_boundaries({'a0': {'min': 0.},
                            'a1': {'max': 0.},
                            'A': {'min': 0.},
                            'FWHM': {'min': 0.},
                            'mu': {'min': 0.}})

    success, message = sat.chisquare_fit(model, fitDF.frequency.to_numpy(), fitDF.intensity.to_numpy(), yerr=fitDF.intensityUnc.to_numpy())
    modelDF = model.get_result_frame()

    binPerHz = len(fitDF)/(fitDF.frequency.max()-fitDF.frequency.min())
    diff = fLarmor - ufloat(modelDF.mu.Value.values[0], modelDF.mu.Uncertainty.values[0],)
    maxInt = model(fitDF.frequency.values[0])

    # fit5sDF = fitDF.loc[(fitDF['frequency'] < upper5s)].std()

    # if modelDF.FWHM.Value.values[0] > 120.:
    #     lower5s = modelDF.mu.Value.values[0]-6*modelDF.FWHM.Value.values[0]
    #     upper5s = modelDF.mu.Value.values[0]+6*modelDF.FWHM.Value.values[0]
    #     fit5sDF = fitDF.loc[(fitDF['frequency'] > lower5s) & (fitDF['frequency'] < upper5s)].reset_index(drop = True)
    #     A5s = fit5sDF.intensity.sum()
    #     dA5s = A5s**0.5
    # else:
    #     A5s = np.nan
    #     dA5s = np.nan


    if isotope != 'H1':
        print('con')

    A3s = np.nan
    dA3s = np.nan
    resultDF = pd.DataFrame([[A3s, dA3s,
        modelDF.A.Value.values[0]*binPerHz, modelDF.A.Uncertainty.values[0]*binPerHz,
        modelDF.mu.Value.values[0], modelDF.mu.Uncertainty.values[0],
        modelDF.FWHM.Value.values[0], modelDF.FWHM.Uncertainty.values[0],
        modelDF.a0.Value.values[0], modelDF.a0.Uncertainty.values[0],
        modelDF.a1.Value.values[0], modelDF.a1.Uncertainty.values[0],
        fLarmor.nominal_value, fLarmor.std_dev, diff.nominal_value, diff.std_dev, maxInt,
        modelDF.Chisquare.values[0], modelDF.NDoF.values[0], float(modelDF.Chisquare.values[0]/modelDF.NDoF.values[0])]],
        columns=['A3s', 'dA3s','A', 'dA',
                'mu', 'dmu', 'FWHM', 'dFWHM', 
                'a0', 'da0', 'a1', 'da1',
                'fLarmor', 'dFlarmor', 'diffLarmor', 'ddiffLarmor', 'maxIntensity',
                'Chi2', 'NDoF', 'Red. Chi2'])

    resultDF = resultDF.reset_index(drop = True)
    resultDF.to_csv(resultFile, index = False)

    plt.figure(dpi = 200)
    plt.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensityUnc), label = "FFT", fmt='.', zorder=5)
    plt.plot(fitDF.frequency.to_numpy(), model(fitDF.frequency.to_numpy()), label = "Fit", linestyle = "-", zorder=3)
    plt.plot([fLarmor.nominal_value, fLarmor.nominal_value], [fitDF.intensity.min(), fitDF.intensity.max()], label = '$f_{L}$', linestyle='--', zorder=1)
    plt.plot([modelDF.mu.Value.values[0], modelDF.mu.Value.values[0]], [fitDF.intensity.min(), fitDF.intensity.max()], label = 'Fit CoG', linestyle='--', zorder=1)
    plt.legend(loc='upper right')
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


