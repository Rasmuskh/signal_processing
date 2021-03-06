import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns#; sns.set(color_codes=True)
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.signal import convolve


fontsize = 10
#D = pd.read_parquet('../../data/finalData/finalData.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'pred', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<610')
D = pd.read_parquet('../../cnn_test1hour.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 50<=amplitude<6100')

#D['qdc_lg'] = D['qdc_lg_fine']
#D['qdc_sg'] = D['qdc_sg_fine']
#D['ps'] = D['ps_fine']
fsg=4900
flg=250
D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_D.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3



def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
# k=7
# kernel = [0]*k
# a, x0, sigma = 1, 4, 1
# for i in range(0,7):
#     kernel[i]=gaus(i+1, a, x0, sigma)
# kernel=np.array(kernel)
# kernel=kernel/sum(kernel)
# t = np.linspace(0,205,206)
# H = np.histogram(D.tof, bins =200, range=(-50,150))
# H = convolve(H[0], kernel)
# plt.figure(figsize=(6.2,2.1))
# plt.plot(t, H, color='black')
# plt.gca().fill(t[0:205], H[0:205], "orange", alpha=0.8)


# k=40
# plt.gca().fill(t[10:24]+k, H[10+k:24+k], "blue", alpha=0.8)
# plt.gca().fill(t[42:74], H[42+k:74+k], "red", alpha=0.8)
# plt.box(on=None)
# plt.xlim(5,100)
# #textstr='Sketch of a Time of Flight Spectrum'
# #plt.text(30, 460, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', linewidth=1.5, edgecolor='black', pad=0.5, boxstyle='square'))
# ax = plt.gca()
# #ax.annotate('$\gamma, n$', xy=(55, 150), xytext=(72 ,300), fontsize= fontsize, arrowprops=dict(facecolor='black', shrink=0.05, width=1, frac=0.10, headwidth=9),)
# #ax.annotate('$\gamma, \gamma$', xy=(15, 150), xytext=(30,300), fontsize= fontsize, arrowprops=dict(facecolor='black', shrink=0.05, width=1, frac=0.10, headwidth=9),)

# plt.xlim(-10,155)
# plt.xlim(-10,200)
# #plt.xticks([], [])
# #plt.yticks([], [])
# plt.ylabel('Counts')
# plt.xlabel('Time(ns)')
# plt.show()


dummy=D.query('0<ps<0.5 and 0<E<5').head(10000)
plt.figure(figsize=(6.2,3.1))
plt.subplot(1,2,1)
plt.scatter(dummy.E, 1-(dummy.qdc_sg/ (dummy.qdc_lg+5000)), color='black', alpha=0.05, s=10)

plt.box(on=None)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel('$\mathrm{Q_{lg}}$')
plt.ylabel('PS')
plt.subplot(1,2,2)
dummyN=dummy.query('ps>=0.222')
plt.scatter(dummyN.E, dummyN.ps, color='red', alpha=0.1, s=10)
dummyY=dummy.query('ps<0.222')
plt.scatter(dummyY.E, dummyY.ps, color='blue', alpha=0.1, s=10)
plt.axhline(y=0.222, linestyle='--', color='black')
plt.box(on=None)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel('$\mathrm{Q_{lg}}$')
plt.ylabel('PS')
plt.tight_layout()
plt.show()
