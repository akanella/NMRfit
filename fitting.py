import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import satlas as sat
from uncertainties import ufloat
import uncertainties.unumpy as unp

magneticField = 4.48 # in mT
isotope = 'Xe129' # choose between H1, H2, Xe129, Xe131, Xe133, Xe129m, Xe131m, Xe133m

# main_dir = 'C:\\Users\\quentin.rogliard\\OneDrive - HESSO\\Documents\\GitHub\\NMRfit'
main_dir = '/Users/akanellako/Documents/NMR_data'

# fileNames = ['fft-one_shot_0s_laseroff_120deg_full_polarisation.txt',
#                 'fft-one_shot_10s_laseroff_120deg_full_polarisation.txt',
#                 'fft-one_shot_30s_laseroff_120deg_full_polarisation.txt',
#                 'fft-one_shot_60s_laseroff_120deg_full_polarisation.txt',
#                 'fft-one_shot_90s_laseroff_120deg_full_polarisation.txt',
#                 'fft-one_shot_120s_laseroff_120deg_full_polarisation.txt']

def gyromagneticRatio(isotope):
    if isotope == 'H1':
        magneticMoment = ufloat(2.792847351, 0.000000009)  # in nuclear magnetons
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

    return gyromagneticRatio, nucSpin

def larmorFrequency(magneticField, isotope):

    larmorFrequency = gyromagneticRatio(isotope)[0] * ufloat(magneticField, 0.005)

    return abs(larmorFrequency)

def sphereVolume(radius):

    volume = (4/3) * math.pi * (radius**3)

    return volume

def nbAtomsIdealGas(pressure, volume, temperature):

    nbAtoms = 1e2 * 1e-6 * pressure * volume / (ufloat(cts.physical_constants['Boltzmann constant'][0], cts.physical_constants['Boltzmann constant'][2]) * temperature)

    return nbAtoms

def protonThermalPolarisation(magneticField, temperature):

    protonMagneticMoment = ufloat(1.41060679545e-26, 0.00000000060e-26) # in J/T
    polarisation = (protonMagneticMoment * ufloat(magneticField, 0.005) * 1e-3) / (ufloat(cts.physical_constants['Boltzmann constant'][0], cts.physical_constants['Boltzmann constant'][2]) * temperature)
    polarisation = unp.tanh(polarisation)

    return polarisation

def NMRPeakModel(x, par):

    A = par[0]
    mu = par[1]
    gamma = par[2]
    a0 = par[3]
    a1 = par[4]

    linear = (a1 * x) + a0

    lorentzian = (1/math.pi) * A * ((gamma) / (((x - mu)**2) + ((gamma)**2)))


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

    resultFile = os.path.join(resultDir, '{}{}.csv'.format(fileName[:-4], '{}'))
    walkFile = os.path.join(resultDir, '{}.h5'.format(fileName[:-4]))
    resultGraph = os.path.join(resultDir, '{}{}.png'.format(fileName[:-4], '{}'))

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
    fitDF.to_csv(resultFile.format('_spectrum'), index = False)
    modelFrequency = np.arange(fitDF.frequency.min(), fitDF.frequency.max()+50, 50)

    # Initialisation of model parameters
    init_mu = fLarmor.nominal_value
    init_gamma = 120.
    init_A = inDF.intensity.max()*2*init_gamma*1e-3
    init_a1 = (fitDF.intensity.iloc[-1]-fitDF.intensity.iloc[0])/(fitDF.frequency.iloc[-1]-fitDF.frequency.iloc[0])
    init_a0 = inDF.intensity.min()
    init = [init_A, init_mu, init_gamma, init_a0, init_a1]

    model = sat.MiscModel(NMRPeakModel, init, ['A', 'mu', 'gamma', 'a0', 'a1'])
    model.set_boundaries({'a0': {'min': 0.},
                            'A': {'min': 0.},
                            'gamma': {'min': 20.},
                            'mu': {'min': 0.}})

    plt.figure(dpi = 200)
    plt.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensityUnc), label = 'data', fmt='.', zorder=5)
    plt.plot(modelFrequency, model(modelFrequency), label = 'init fit', linestyle = '-', zorder=3)
    plt.legend(loc='upper right')
    plt.xlabel('Frequence [Hz]')
    plt.ylabel('Intensity [a.u.]')
    # plt.show()
    plt.close()

    success, message = sat.chisquare_fit(model, fitDF.frequency.to_numpy(), fitDF.intensity.to_numpy(), yerr=fitDF.intensityUnc.to_numpy())
    model.display_chisquare_fit()

    band = sat.create_band(model, x=modelFrequency, x_data=fitDF.frequency, y_data=fitDF.intensity, yerr=fitDF.intensityUnc, xerr=None, method='chisquare', kind='confidence')
    residuals = (fitDF.intensity.values - model.seperate_response(fitDF.frequency.values)) / fitDF.intensityUnc.values
    residualsMean = ufloat(residuals[0].mean(), residuals[0].std())

    # success, message = sat.likelihood_fit(model, fitDF.frequency.to_numpy(), fitDF.intensity.to_numpy(), hessian=False, walking=True, walk_kws={'filename':walkFile, 'nsteps': 1e3, 'walkers': 60})
    # model.display_chisquare_fit()

    # sat.process_walk(model, walkFile)
    # fig, ax, cb = sat.generate_correlation_plot(walkFile)
    # fig, ax = sat.generate_walk_plot(walkFile, selection=[1,100])
    # fig.savefig(resultGraph.format('_walk'))

    # fig, ax, cb = sat.generate_correlation_map(model, fitDF.frequency.tolist(), fitDF.intensity.tolist(), method='chisquare', distance=5)
    # fig.savefig(resultGraph.format('correlation'))

    modelDF = model.get_result_frame()

    binPerHz = len(fitDF)/(fitDF.frequency.max()-fitDF.frequency.min())
    diff = fLarmor - ufloat(modelDF.mu.Value.values[0], modelDF.mu.Uncertainty.values[0])
    maxInt = model(fitDF.frequency.values[0])
    
    #Polarisation Calculation
    if isotope != 'H1':
        waterMolarMass = ufloat(18.01528, 0.00033)
        pressure = 0.264 * ufloat(100, 10) # in mbar
        temperature = ufloat(300, 3) # in Kelvin
        volume = sphereVolume(ufloat(1.1, 0.005)) # in cm2, radius in cm
        polarisation_H1 = protonThermalPolarisation(1.25, temperature)
        A_H1 = ufloat(np.mean([1296.2079525436134, 1208.850679785642]),np.std([1296.2079525436134, 1208.850679785642]))
        gyromagneticRatio_H1 = gyromagneticRatio('H1')[0].nominal_value
        nbAtoms_H1 = 2 * volume * (1/waterMolarMass) * ufloat(cts.physical_constants['Avogadro constant'][0], cts.physical_constants['Avogadro constant'][2])
        nucSpin_H1 = gyromagneticRatio('H1')[1]
        gyromagneticRatio_isotope = gyromagneticRatio(isotope)[0].nominal_value
        nbAtoms_isotope = nbAtomsIdealGas(pressure, volume, temperature)
        nucSpin_isotope = gyromagneticRatio(isotope)[1]

        polarisation = polarisation_H1 * (modelDF.A.Value.values[0]/A_H1) * (nbAtoms_H1 / nbAtoms_isotope) * (gyromagneticRatio_H1 / gyromagneticRatio_isotope) * (nucSpin_H1 / nucSpin_isotope) 
        polarisation = abs(polarisation)
    else:
        polarisation = ufloat(np.nan, np.nan)

    if 2*modelDF.gamma.Value.values[0] > 120.:
        lower3s = modelDF.mu.Value.values[0]-6*modelDF.gamma.Value.values[0]
        upper3s = modelDF.mu.Value.values[0]+6*modelDF.gamma.Value.values[0]
        fit3sDF = fitDF.loc[(fitDF['frequency'] > lower3s) & (fitDF['frequency'] < upper3s)].reset_index(drop = True)
        A3s = fit3sDF.intensity.sum()
        dA3s = A3s**0.5
    else:
        A3s = np.nan
        dA3s = np.nan

    resultDF = pd.DataFrame([[magneticField, isotope, gyromagneticRatio(isotope)[1],
        gyromagneticRatio(isotope)[0].nominal_value, gyromagneticRatio(isotope)[0].std_dev,
        A3s, dA3s,
        modelDF.A.Value.values[0]*binPerHz, modelDF.A.Uncertainty.values[0]*binPerHz,
        modelDF.mu.Value.values[0], modelDF.mu.Uncertainty.values[0],
        2*modelDF.gamma.Value.values[0], 2*modelDF.gamma.Uncertainty.values[0],
        modelDF.a0.Value.values[0], modelDF.a0.Uncertainty.values[0],
        modelDF.a1.Value.values[0], modelDF.a1.Uncertainty.values[0],
        fLarmor.nominal_value, fLarmor.std_dev, diff.nominal_value, diff.std_dev, maxInt, polarisation.nominal_value, polarisation.std_dev,
        modelDF.Chisquare.values[0], modelDF.NDoF.values[0], float(modelDF.Chisquare.values[0]/modelDF.NDoF.values[0])]],
        columns=['Bo', 'isotope', 'I', 'gamma/2pi', 'dgamma/2pi', 
                'A3s', 'dA3s','A', 'dA',
                'mu', 'dmu', 'FWHM', 'dFWHM', 
                'a0', 'da0', 'a1', 'da1',
                'fLarmor', 'dfLarmor', 'diffLarmor', 'ddiffLarmor', 'maxIntensity', 'polarisation', 'dpolarisation',
                'Chi2', 'NDoF', 'Red. Chi2'])

    resultDF = resultDF.reset_index(drop = True)
    resultDF.to_csv(resultFile.format(''), index = False)

    plt.figure(dpi = 200)
    plt.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensityUnc), label = 'FFT', fmt='.', zorder=5)
    plt.plot(modelFrequency, model(modelFrequency), label = 'Fit', linestyle = '-', zorder=3)
    plt.fill_between(modelFrequency, model(modelFrequency)-band, model(modelFrequency)+band, alpha=0.33, color='tab:orange')
    plt.axvline(fLarmor.nominal_value, label = '$f_{L}$', linestyle='--', color='tab:green', zorder=1)
    plt.axvline(modelDF.mu.Value.values[0], label = 'Fit CoG', linestyle='--', color='tab:red', zorder=1)
    plt.legend(loc='upper right')
    plt.xlabel('Frequence [Hz]')
    plt.ylabel('Intensity [a.u.]')
    plt.savefig(resultGraph.format('_simple'))
    plt.close()

    fig, ax = plt.subplots(1, 1, dpi=200)
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan=4)
    ax2 = plt.subplot2grid((5,1), (4,0), sharex=ax1)
    ax1.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensityUnc), label = 'FFT', fmt='.', zorder=5)
    ax1.plot(modelFrequency, model(modelFrequency), label = 'Fit', linestyle = '-', zorder=3)
    ax1.fill_between(modelFrequency, model(modelFrequency)-band, model(modelFrequency)+band, alpha=0.33, color='tab:orange')
    # ax1.axvline(fLarmor.nominal_value, label = '$f_{L}$', linestyle='--', color='tab:green', zorder=1)
    # ax1.axvline(modelDF.mu.Value.values[0], label = 'Fit CoG', linestyle='--', color='tab:red', zorder=1)
    # ax1.tick_params(axis='x', which='both', length=0)
    ax1.set_ylabel('Intensity [a.u.]')
    ax1.legend()
    # ---
    ax2.plot(fitDF.frequency, residuals[0], '.', color='k')
    ax2.axhline(residualsMean.nominal_value, label='{:.1f}$\pm${:.1f} $\sigma$'.format(residualsMean.nominal_value, residualsMean.std_dev), color='red', linestyle='--')
    ax2.set_xlabel('Frequence [Hz]')
    ax2.set_ylabel('Residuals [$\sigma$]')
    ax2.legend()
    plt.xlim([50000, 56000])
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0)
    plt.savefig(resultGraph.format('_residuals'))
    plt.close()

    return fitDF

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

if len(fileNames) > 1:
    i = 0
    plt.figure(dpi = 200)

    for fileName in fileNames:
            
            mainDF = main(data_dir, result_dir, fileName)

            plt.errorbar(mainDF.frequency, mainDF.intensity, yerr=(mainDF.intensityUnc), label=fileName, fmt='.-')

            i+=1

    plt.legend(loc='upper right')
    plt.xlabel('Frequence [Hz]')
    plt.ylabel('Intensity [a.u.]')
    plt.xlim([50000, 56000])
    plt.savefig(result_dir + '/all.png')

else:
    fileName = fileNames[0]
    main(data_dir, result_dir, fileName)



