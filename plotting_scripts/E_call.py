import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

import seaborn as sns; sns.set(color_codes=True, style='ticks')
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from math import log
import sys
sys.path.append('../tof')
sys.path.append('../../analog_tof')
import pyTagAnalysis as pta

fontsize=10
colorlist=['red', 'purple', 'orange']
print("chose analog or digital data: \"A/D\"")
mode = input()
if mode == "A" or mode == "a":
    mode = 'A'
    N = pta.load_data('../data/finalData/Data1793_cooked.root')
    N['qdc_lg'] = N['qdc_det0']

    minimum, maximum = -100, 5000
    p1, p2 = 60, 70
    p3, p4 = 1100, 1600
    p5, p6 = 2500, 3100
elif mode == "D" or mode == "d":
    mode = 'D'
    N = pd.read_parquet('../data/finalData/finalData.pq', engine='pyarrow', columns=['pred', 'cfd_trig_rise', 'window_width', 'tof', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps', 'fine_baseline_offset']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<amplitude<6100')
    N['qdc_lg'] = (N['qdc_lg']+500*N['fine_baseline_offset'])/1000
    C  = pd.read_parquet('../data/finalData/specialdata/cobalt60_5min.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'qdc_lg', 'qdc_sg', 'ps']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<6100')
    C['qdc_lg'] = C['qdc_lg_fine']/1000
    minimum, maximum = -2000, 60000
    p1, p2 = 3200, 5000
    p3, p4 = 6200, 8900
    p5, p6 = 14500, 17000
def getBinCenters(bins):
    """ From Hanno Perrey calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0,len(bins)-1)])

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
def get_ADC_error(x0, sigma, dx0,dsigma, frac):
    return ( dx0**2 + (sigma**2)*4*(log(frac))**2  )**(1/2)

def get_ADC_channel(x0, sigma, frac):
    return x0 + sigma*(-2*log(0.89) )**(1/2)

def fit_gaus(left, right, df):
    #x = np.linspace(left, right, int(0.5+(right-left) + 1))
    #H = np.histogram(df.qdc_lg, range=(minimum, maximum), bins=(maximum-minimum))
    #y = H[0][int(left+0.5):int(right+0.5)+1]
    H = np.histogram(df.qdc_lg, range=(left, right), bins=(right-left))
    x = getBinCenters(H[1])
    y=H[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 100])
    adc_err = get_ADC_error(x0=popt[1], sigma=popt[2], dx0=pcov[1][1], dsigma=pcov[2][2], frac=0.89)
    adc89 = get_ADC_channel(x0=popt[1], sigma=popt[2], frac=0.89)
    return popt, pcov, adc_err, adc89


if mode == 'A':
    popt_1, pcov_1, adc_err_1, p89_1 = fit_gaus(p1, p2, N)
elif mode == 'D':
    popt_1, pcov_1, adc_err_1, p89_1 = fit_gaus(p1, p2, C)

popt_2, pcov_2, adc_err_2, p89_2 = fit_gaus(p3, p4, N)
popt_3, pcov_3, adc_err_3, p89_3 = fit_gaus(p5, p6, N)
errList = np.array([adc_err_1, adc_err_2, adc_err_3])


#Plot stuff
plt.figure(figsize=(6.2,5))
#plot raw qdc spectrum
ax1 = plt.subplot(2, 1, 1)
l1 = minimum; l2 = maximum
b =  int((maximum-minimum)/10)
fac = (l2-l1)/b
plt.hist(N.qdc_lg, bins=b, range=(l1,l2), histtype='step', lw=1, log=True, zorder=1, label='Pu/Be')
if mode == "D":
    plt.hist(C.qdc_lg, bins=b, range=(l1,l2), histtype='step', lw=1, log=True, zorder=2, label='$^{60}$Co')
#plot gaussian fits
x = np.linspace(p1, p2, (p2-p1)*1)
if mode == "A":
        lbl = "pedestal"
else:
        lbl = "1.33 MeV"
if mode == 'D':
    plt.plot(x, fac*gaus(x, popt_1[0], popt_1[1], popt_1[2]), ms=6, zorder=4, label=lbl, color=colorlist[0], linestyle=':')
x = np.linspace(p3, p4, (p4-p3)*1)
plt.plot(x, fac*gaus(x, popt_2[0], popt_2[1], popt_2[2]), ms=6, zorder=4, label='2.23 MeV', color=colorlist[1], linestyle='-')
x = np.linspace(p5, p6, (p6-p5)*1)
plt.plot(x, fac*gaus(x, popt_3[0], popt_3[1], popt_3[2]), ms=6, zorder=5, label='4.44 MeV', color=colorlist[2], linestyle='--')
plt.legend(loc='upper right')
if mode == 'A':
    plt.xlabel('QDC channel', fontsize=fontsize)
else:
    plt.xlabel('Digitizer pulse integration', fontsize=fontsize)
plt.ylabel('Counts', fontsize=fontsize)
plt.ylim(0.1, 3*10**5)
plt.xlim(minimum, maximum)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)


#fit a line through the two known energies
ax2=plt.subplot(2, 1, 2)
E1 = 1.17
E2 = 2.23
E3 = 4.44

EeMax1 = 2*E1**2/(0.511+2*E1)
EeMax2 = 2*E2**2/(0.511+2*E2)
EeMax3 = 2*E3**2/(0.511+2*E3)
if mode=="A":
    Elist = np.array([0, EeMax2, EeMax3])#, EeMax4]
    qdclist = np.array([popt_1[1], p89_2, p89_3])#, p89_4]
    labellist=['pedestal', '2.23 MeV', '4.44 MeV']
else:
    Elist = np.array([EeMax1, EeMax2, EeMax3])#, EeMax4]
    qdclist = np.array([p89_1, p89_2, p89_3])#, p89_4]
    labellist=['1.33 MeV', '2.23 MeV', '4.44 MeV']
def lin(x, a, b):
    return a*x +b
#popt,pcov = curve_fit(lin, qdclist, Elist, p0=[1, 0])
#dev = np.sqrt(np.diag(pcov))

popt_lin = np.polyfit(qdclist, Elist, deg=1, w=1/errList)

x = np.linspace(minimum, int(max(qdclist)*1.1), 2000)
plt.plot(x, x*popt_lin[0]+popt_lin[1], lw=1.5)
for i in range(0, 3):
    plt.errorbar(qdclist[i], Elist[i], color=colorlist[i], xerr=errList[i], fmt='o', ms=3, lw=1.5, label = labellist[i])
plt.legend(frameon=True)
plt.xlabel('QDC bin', fontsize=fontsize)
plt.ylabel('MeV$_{ee}$', fontsize=fontsize)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)



# ax3=plt.subplot(2, 2, 3)
# plt.hist(popt_lin[1] + N.qdc_lg*popt_lin[0], bins= int((maximum-minimum)/10), range=(popt_lin[1] + (minimum)*popt_lin[0], popt_lin[1] + (maximum)*popt_lin[0]), histtype='step', log=True, lw=1, label='Calibrated energy spectrum')
# plt.xlabel('MeV$_{ee}$', fontsize=fontsize)
# plt.ylabel('Counts', fontsize=fontsize)
# plt.ylim(0.1, 10**5)
# plt.xlim(popt_lin[1] + (minimum)*popt_lin[0], popt_lin[1] + (maximum)*popt_lin[0])

# ax = plt.gca()
# ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
# plt.legend(frameon=True)


#add the calibrated axis!
ax4 = ax1.twiny()
ax4.set_xlim(popt_lin[1]+ (minimum)*popt_lin[0], popt_lin[1]+ (maximum)*popt_lin[0] )
plt.xlabel('MeV$_{ee}$', fontsize=fontsize)
plt.tight_layout()

if mode == "A":
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/Ecall.pdf', format='pdf')
else:
    plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/Ecall.pdf', format='pdf')
plt.show()


