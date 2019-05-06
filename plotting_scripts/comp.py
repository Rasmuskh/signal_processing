# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True, style='ticks')
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
A = pta.load_data('../data/finalData/Data1793_cooked.root').query('0<qdc_det0<50000')
A['tof'] = 1000 - A['tdc_det0_yap0']
acal = np.load('../data/finalData/Ecal_A.npy')
A['E'] = acal[1]+acal[0]*A.qdc_det0

def gaus(x, a, x0, sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
def fit_gaus(left, right, df):
    x = np.linspace(left, right, (right- left)*4)
    H = np.histogram(df.tof, range=(left, right), bins=(right-left)*4)
    #plt.step(x,H[0])
    y = H[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[max(y), np.mean(x), 5])
    #plt.plot(x, gaus(x, popt[0], popt[1], popt[2]))
    #plt.show()
    return popt, pcov

def plot_E_comp(df_D, df_A, w_A, w_D):
    thrList=[50, 155]
    d = df_D.query('amplitude>%s'%(thrList[0]))
    plt.hist(d.E, weights=[1]*len(d), bins=150, range=(0,6), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digital setup\nRaw: %.2fx$10^6$ events'%(len(df_D)/10**6))
    d = df_D.query('amplitude>%s'%(thrList[1]))


    plt.hist(d.E, weights=[w_D]*len(d), bins=150, range=(0,6), log=True, histtype='step', alpha=0.75, lw=1.5, label='Digital setup\nNormalized')
    plt.hist(df_A.E, weights=[w_A]*len(df_A), bins=150, range=(0,6), log=True, histtype='step', alpha=0.75, lw=1.5, label='Analog setup\nLivetime corrected: %.2fx$10^6$ events'%(w_A*len(df_A)/10**6))
    plt.xlabel('Energy $\mathrm{MeV}_{ee}$', fontsize=fontsize)
    plt.ylabel('counts', fontsize=fontsize)
    plt.legend(loc=8)
    plt.ylim(1,300000)
    #if textstr:
    #    plt.text(1, 3000, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)

def plot_tof_comp(df_D, df_A, w_A, w_D, thr, mode, tlim=(-20,80), textstr=''):

    #Digitized
    dlim=[0, 200]
    dummy_D = df_D.query('%d<=tof<%d'%(dlim[0], dlim[1]))
    popt_D, pcov_D = fit_gaus(15, 35, dummy_D)
    print(popt_D[1])
    dummy_D['tof'] = (D['tof'] - popt_D[1])
    dummy_D = dummy_D.query('%s<tof<%s'%(tlim[0], tlim[1]))
    #analog
    Tcal = np.load('../data/finalData/T_cal_analog.npy')
    alim=[-1000, 1000]
    dummy_A = df_A.query('qdc_det0>500 and %d<=tof<%d'%(alim[0], alim[1]))
    popt_A, pcov_A = fit_gaus(350, 400, dummy_A)
    print(popt_A[1])
    dummy_A['tof'] = (A['tof'] - popt_A[1])*(-Tcal[0])
    dummy_A = dummy_A.query('%s<tof<%s'%(tlim[0], tlim[1]))
    dummy_D = dummy_D.query('amplitude>%s'%thr)
    d_detect = 1.055+0.005+179/1000/2
    if mode == 'unadjusted':
            plt.hist(dummy_D.tof + d_detect/c, weights=[w_D]*len(dummy_D), bins=tlim[1]-tlim[0], range=(tlim[0],tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Digital setup\nUnadjusted: %.1fx$10^3$ events'%(w_D*len(dummy_D)/1000), color='blue')
            plt.hist(dummy_A.tof + d_detect/c, weights=[w_A]*len(dummy_A), bins=tlim[1]-tlim[0], range=(tlim[0], tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Analog setup\nUnadjusted: %.1fx$10^3$ events'%(w_A*len(dummy_A)/1000), color='red')
    elif mode == 'adjusted':
            plt.hist(dummy_D.tof + d_detect/c, weights=[w_D]*len(dummy_D), bins=tlim[1]-tlim[0], range=(tlim[0],tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Digital setup\nNormalized and threshold adjusted', color='blue')
            plt.hist(dummy_A.tof + d_detect/c, weights=[w_A]*len(dummy_A), bins=tlim[1]-tlim[0], range=(tlim[0], tlim[1]), histtype='step', alpha=0.75, lw=1.5, label='Analog setup\nLivetime corrected: %.1fx$10^3$ events'%(w_A*len(dummy_A)/1000), color='red')
    plt.xlabel('ToF(ns)', fontsize=fontsize)
    plt.ylabel('counts', fontsize=fontsize)
    plt.legend(loc=0)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    ylim = plt.gca().get_ylim()
    plt.ylim(0,1.1*ylim[1])
    ylim = plt.gca().get_ylim()
    py = (ylim[1]-ylim[0])*0.9
    if textstr:
        plt.text(60, py, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    return (popt_A, popt_D)

def getBinCenters(bins):
    """ From Hanno Perrey calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0, len(bins)-1)])
def plot_qdc_ratio(df_D, df_A, w_A, w_D, textstr=''):
    H_D = np.histogram(df_D.E, weights=[w_D]*len(df_D), bins=150, range=(0,6))
    H_A = np.histogram(df_A.E, weights=[w_A]*len(df_A), bins=150, range=(0,6))

    plt.plot(getBinCenters(H_D[1]), H_D[0]/H_A[0], label='Count ratio, Digital/Analog')
    plt.grid()
    plt.axhline(y=1, linestyle='--', color='black')
    loc = plticker.MultipleLocator(base=0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(loc)

    if textstr:
        plt.text(1, 4, textstr, fontsize = fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    plt.xlabel('Energy $\mathrm{MeV}_{ee}$', fontsize=fontsize)
    plt.ylabel('Ratio of counts', fontsize=fontsize)
    plt.legend(frameon=True)






w_A = 1/0.4432267926625903
w_D = max(np.histogram(A.E, bins=200, weights=[w_A]*len(A), range=(0,8))[0])/max(np.histogram(D.query('amplitude>155').E, bins=200, range=(0,8))[0])

# #figure size
# plt.figure(figsize=(6.2, 6))
# plt.subplot(2,1,1)
# #===QDC Spectra===
# plot_E_comp(df_D=D, df_A=A, w_A=w_A, w_D=w_D)
# plt.tight_layout()

# plt.subplot(2,1,2)
# #figure size
# plot_qdc_ratio(df_D=D.query('amplitude>155'), w_A=w_A, w_D=w_D, df_A=A, textstr='Livetime adjusted\nDigital threshold =  %.0f mV'%(155*1000/1024))
# plt.tight_layout()
# plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/qdc_comp.png', format='png')
# plt.show()


A = A.query('500<qdc_det0<5000')
#figure size
plt.figure(figsize=(6.2, 5))
plt.subplot(2,1,1)
thr=50
popt_a, popt_d = plot_tof_comp(df_D=D, df_A=A, w_A=1, w_D=1, thr=thr, mode='unadjusted')
fwhm_D1 = 2*(2*np.log(2))**(1/2)*popt_d[2]
fwhm_A1 = 2*(2*np.log(2))**(1/2)*popt_a[2]

plt.subplot(2,1,2)
D = pd.read_parquet('../yapThres45.pq/', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'invalid']).query('channel==0 and invalid==False and %s<=amplitude<6180'%Dthres)
dcal = np.load('../data/finalData/Ecal_D.npy')
D['E'] = dcal[1]+dcal[0]*D.qdc_lg/1000
D['tof'] = D['tof']/1000
w_D = max(np.histogram(A.E, bins=200, weights=[w_A]*len(A), range=(0,8))[0])/max(np.histogram(D.query('amplitude>155').E, bins=200, range=(0,8))[0])

thr=150
popt_a, popt_d = plot_tof_comp(df_D=D, df_A=A, w_A=w_A, w_D=w_D, thr=thr, mode='adjusted')
fwhm_D2 = 2*(2*np.log(2))**(1/2)*popt_d[2]
fwhm_A2 = 2*(2*np.log(2))**(1/2)*popt_a[2]
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/CompareResults/tof_comp.pdf', format='pdf')
plt.show()

