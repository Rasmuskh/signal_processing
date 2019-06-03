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
D = pd.read_parquet('../../data/finalData/finalData.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg_fine', 'qdc_sg_fine', 'ps_fine', 'pred', 'qdc_lg', 'qdc_sg', 'ps', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<610')

Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_D.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 3.3


fsg=0
flg=0
for i in range(0, 101):
    fsg = i*49
    flg = i*2.5
    D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
    dummy=D.query('0<ps<1 and 0<E<5')#.head(24000)
    #plt.scatter(dummy.E, dummy.ps, color='black', alpha=0.05, s=10, label='a=%s\nb=%s'%(fsg, flg))
    g = sns.jointplot(ratio=3,kind='hex', x=dummy.E, y=dummy.ps, xlim=(0, 5), ylim=(0, 0.5), stat_func=None, cmap='viridis', gridsize=(100,100), vmin=0, vmax=1200)
    textstr = 'a=%.2f V$\cdot$ns\nb=%.2f V$\cdot$ns'%(round(fsg/1000,6), round(flg/1000,6))
    g.ax_joint.text(3, 0.45, textstr, color='white', fontsize=12, verticalalignment='top')
    plt.hist(dummy.ps, bins=100, range=(0,1), label='a=%s\nb=%s'%(fsg, flg))
    plt.ylabel('PS', fontsize=12)
    plt.xlabel('$E\mathrm{(MeV_{ee})}$', fontsize=12)
    plt.tight_layout()
    plt.savefig('hex/psd%s.png'%i, format='png')
    plt.close()


fsg = 4900
flg = 250
D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
dummy=D.query('0<ps<1 and 0<E<5')#.head(24000)
#plt.scatter(dummy.E, dummy.ps, color='black', alpha=0.05, s=10, label='a=%s mV$\cdot$ns\nb=%s mV$\cdot$ns'%(fsg, flg))
g = sns.jointplot(ratio=3,kind='hex', x=dummy.E, y=dummy.ps, xlim=(0, 5), ylim=(0, 0.5), stat_func=None, cmap='viridis', gridsize=(100,100), vmin=0, vmax=1200)
textstr = 'a=%.2f V$\cdot$ns\nb=%.2f V$\cdot$ns'%(round(fsg/1000,6), round(flg/1000,6))
g.ax_joint.text(3, 0.45, textstr, color='white', fontsize=12, verticalalignment='top')
g.ax_joint.axhline(y=0.222, linestyle='--', color='white', lw=1)
plt.hist(dummy.ps, bins=100, range=(0,1), label='a=%s\nb=%s'%(fsg, flg))
plt.ylabel('PS', fontsize=12)
plt.xlabel('$E\mathrm{(MeV_{ee})}$', fontsize=12)
plt.tight_layout()
plt.savefig('hex/psd%s.png'%'_cut', format='png')
plt.show()

