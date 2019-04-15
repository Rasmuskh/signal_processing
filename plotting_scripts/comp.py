# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True, style='whitegrid')
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta
import matplotlib.ticker as plticker


c=0.299792458# m/ns
fontsize = 10

Dthres=50
D = pd.read_parquet('../data/finalData/finalData.pq/', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'invalid']).query('channel==0 and invalid==False and %s<=amplitude<6180'%Dthres)
dcal = np.load('../data/finalData/Ecal_D.npy')
D['E'] = dcal[1]+dcal[0]*D.qdc_lg/1000
D['tof'] = D['tof']/1000
A = pta.load_data('../data/finalData/Data1793_cooked.root').query('0<qdc_det0<5000')
A['tof'] = 1000 - A['tdc_det0_yap0']
acal = np.load('../data/finalData/Ecal_A.npy')
A['E'] = acal[1]+acal[0]*A.qdc_det0

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
def fit_gaus(left, right, df):
    x = np.linspace(left, right, right- left)
    H = np.histogram(df.tof, range=(left, right), bins=(right-left))
    y = H[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 5])
    return popt, pcov

def plot_E_comp(df_D, df_A, w_A, w_D, textstr=''):
    plt.hist(df_D.E, weights=[w_D]*len(df_D), bins=200, range=(0,8), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digital setup: %.2fx$10^6$ events'%(w_D*len(df_D)/10**6))
    plt.hist(df_A.E, weights=[w_A]*len(df_A), bins=200, range=(0,8), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup: %.2fx$10^6$ events'%(w_A*len(df_A)/10**6))
    plt.xlabel('Energy $MeV_{ee}$', fontsize=fontsize)
    plt.ylabel('counts', fontsize=fontsize)
    plt.legend(loc=8)
    plt.ylim(1,300000)
    if textstr:
        plt.text(1, 3000, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)

def plot_tof_comp(df_D, df_A, w_A, w_D, tlim=(-20,130), textstr=''):

    #Digitized
    dlim=[0, 200]
    dummy_D = df_D.query('%d<=tof<%d'%(dlim[0], dlim[1]))
    popt_D, pcov_D = fit_gaus(15, 35, dummy_D)
    dummy_D['tof'] = (D['tof'] - popt_D[1])
    dummy_D = dummy_D.query('%s<tof<%s'%(tlim[0], tlim[1]))
    #analog
    Tcal = np.load('../data/finalData/T_cal_analog.npy')
    alim=[-1000, 1000]
    dummy_A = df_A.query('qdc_det0>500 and %d<=tof<%d'%(alim[0], alim[1]))
    popt_A, pcov_A = fit_gaus(350, 400, dummy_A)
    dummy_A['tof'] = (A['tof'] - popt_A[1])*(-Tcal[0])
    dummy_A = dummy_A.query('%s<tof<%s'%(tlim[0], tlim[1]))

    #df_D_dummy = df_D.query('%s<tof<%s'%(tlim[0], tlim[1])).reset_index()
    plt.hist(dummy_D.tof + 1.055/c, weights=[w_D]*len(dummy_D), bins=tlim[1]-tlim[0], range=(tlim[0],tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Digital setup: %.2fx$10^3$ events'%(w_D*len(dummy_D)/1000))
    #df_A_dummy = df_A.query('%s<tof<%s'%(tlim[0], tlim[1])).reset_index()
    plt.hist(dummy_A.tof + 1.055/c, weights=[w_A]*len(dummy_A), bins=tlim[1]-tlim[0], range=(tlim[0], tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Analog setup: %.2fx$10^3$ events'%(w_A*len(dummy_A)/1000))
    plt.xlabel('ToF(ns)', fontsize=fontsize)
    plt.ylabel('counts', fontsize=fontsize)
    plt.legend(loc=7)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    ylim = plt.gca().get_ylim()
    py = (ylim[1]-ylim[0])*0.9
    if textstr:
        plt.text(60, py, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    return (popt_A, popt_D)

def getBinCenters(bins):
    """ From Hanno Perrey calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0, len(bins)-1)])
def plot_qdc_ratio(df_D, df_A, w_A, w_D, textstr=''):
    H_D = np.histogram(df_D.E, weights=[w_D]*len(df_D), bins=200, range=(0,8))
    H_A = np.histogram(df_A.E, weights=[w_A]*len(df_A), bins=200, range=(0,8))

    plt.plot(getBinCenters(H_D[1]), H_D[0]/H_A[0], label='Count ratio, Digital/Analog')
    loc = plticker.MultipleLocator(base=1.0)
    ax = plt.gca()
    ax.yaxis.set_major_locator(loc)
    if textstr:
        plt.text(1, 4, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    plt.xlabel('Energy $MeV_{ee}$', fontsize=fontsize)
    plt.ylabel('Ratio of counts', fontsize=fontsize)
    plt.legend()






#figure size
plt.figure(figsize=(6.2, 4.8))
#===QDC Spectra===
ax1=plt.subplot(2, 1, 1)
plot_E_comp(df_D=D, df_A=A, w_A=1, w_D=1, textstr='Unadjusted\nDigital threshold =  %.0f mV'%(Dthres*1000/1024))

ax2=plt.subplot(2, 1, 2)
w_analog = 1/0.4432267926625903
thr_D = 155
d = D.query('amplitude>%s'%thr_D)
w_digital = max(np.histogram(A.E, bins=200, weights=[w_analog]*len(A), range=(0,8))[0])/max(np.histogram(d.E, bins=200, range=(0,8))[0])
plot_E_comp(df_D=d, df_A=A, w_A=w_analog, w_D=w_digital, textstr='Livetime adjusted\nDigital threshold =  %.0f mV'%(thr_D*1000/1024))
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/qdc_comp.pdf', format='pdf')
plt.show()

#figure size
plt.figure(figsize=(6.2, 4.8))
ax1=plt.subplot(2, 1, 1)
popt_a, popt_d = plot_tof_comp(df_D=D, df_A=A, w_A=1, w_D=1, textstr='unadjusted\nDigital threshold =  %.0f mV'%(Dthres*1000/1024))
ax1=plt.subplot(2, 1, 2)
popt_a2, popt_d2 = plot_tof_comp(df_D=d, df_A=A, w_A=w_analog, w_D=w_digital, textstr='Livetime adjusted\nDigital threshold =  %.0f mV'%(thr_D*1000/1024))
fwhm_D = 2*(2*np.log(2))**(1/2)*popt_d2[2]
fwhm_A = 2*(2*np.log(2))**(1/2)*popt_a2[2]
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/tof_comp.pdf', format='pdf')
plt.show()


#figure size
plt.figure(figsize=(6.2,2.4))
plot_qdc_ratio(df_D=d, df_A=A, w_A=w_analog, w_D=w_digital, textstr='Livetime adjusted\nDigital threshold =  %.0f mV'%(thr_D*1000/1024))
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/QDC_ratio.pdf', format='pdf')
plt.show()
