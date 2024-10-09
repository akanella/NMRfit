import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import satlas as sat

fCent = 53e3
fLarmor = 53135


# def linearModel(x, par):

#     x0 = par[0]
#     a0 = par[1]
#     a1 = par[2]

#     linear = (a1*(x-x0))+a0

#     return linear

# def lorentzianModel(x, par):

# 	A = par[0]
# 	mu = par[1]
# 	FWHM = par[2]

# 	lorentzian = A * (FWHM/2) / ( math.pi * ((x - mu)**2) + ((FWHM/2)**2) )

# 	return lorentzian  

def NMRPeakModel(x, par):

    A = par[0]
    mu = par[1]
    FWHM = par[2]
    a0 = par[3]
    a1 = par[4]

    linear = (a1 * x) + a0

    lorentzian = A * (FWHM/2) / ( math.pi * ((x - mu)**2) + ((FWHM/2)**2) )


    return linear + lorentzian

# Replace 'your_file.txt' with the actual file path and specify the delimiter if needed.
inDF = pd.read_csv('fft-TEST5_spherical_129Xe_2min_10pts_4000mV_391us_2.81A.txt', delimiter=' ', header=None)

# Assign column names if required
inDF.columns = ['frequency', 'intensity']
inDF['intensityUnc'] = inDF['intensity']**0.5

fitDF = inDF[(inDF['frequency'] >= fCent-5e3) & (inDF['frequency'] <= fCent+5e3)]  

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
# frame.to_csv('test.csv')

binPerHz = len(fitDF)/(fitDF.frequency.max()-fitDF.frequency.min())

plt.figure(dpi = 200)
plt.errorbar(fitDF.frequency, fitDF.intensity, yerr=(fitDF.intensity)**0.5, label = "FFT", fmt='.', zorder=1)
plt.plot(fitDF.frequency.to_numpy(), model(fitDF.frequency.to_numpy()), label = "Fit", linestyle = "--", zorder=2)
plt.plot([fLarmor, fLarmor], [fitDF.intensity.min(), fitDF.intensity.max()])
plt.xlabel("Frequence [Hz]")
plt.ylabel("Intensity [a.u.]")

plt.show()
