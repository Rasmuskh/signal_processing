# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='ticks')
import numpy as np
from scipy.signal import convolve
from scipy import asarray as ar,exp
import dask.dataframe as dd

D=dd.read_parquet('../../data/finalData/finalData.pq/',engine='pyarrow').head(50000)
N=D.query('pred>0.9 and 0<amplitude<600').reset_index()
D=0
n=np.array([0]*320).astype(np.float64)
L=100
for i in range(0,L):
    s = int(0.5+N.cfd_trig_rise[i]/1000)
    n += N.samples[i][s-20:s+300]/N.amplitude[i]

#ax1=plt.subplot(2,1,1)
#plt.plot(-n/min(n), color='blue', lw=3)
#plt.plot(-y/min(y), color='red', lw=3)

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
kernel = [0]*9
a, x0, sigma = 1, 4, 9
for i in range(0,9):
    kernel[i]=gaus(i+1, a, x0, sigma)
kernel=np.array(kernel)
kernel=kernel/sum(kernel)
#ax2=plt.subplot(2,1,2)
n=convolve(-n/min(n), kernel, method='direct', mode='same')
plt.figure(figsize=(6.2,2.4))

k=1
n0 = n[0:-1:1]
x0=np.linspace(0,k*(len(n0)-1),len(n0))
plt.plot(x0, n0, color='black', lw=2,zorder=1, alpha=0.5, label='analog signal')

k=6
n1 = n[0:-1:k]
x1=np.linspace(0,k*(len(n1)-1),len(n1))
plt.scatter(x1, n1, color='blue', marker='x',zorder=2, label='f=$\mathrm{f_0}$')
#plt.plot(x1, n1, color='blue',zorder=3)

k=12
n2 = n[0:-1:k]
x2=np.linspace(0,k*(len(n2)-1),len(n2))
plt.scatter(x2, n2, color='red', marker='.',zorder=3, s=50, label='f=$\mathrm{f_0}/2$')
#plt.plot(x2, n2, color='red',zorder=3)

plt.xlim(0,200)

plt.legend(fontsize=12)
#plt.gca().axes.get_yaxis().set_visible(False)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.xlabel('Time (ns)', fontsize=12)
plt.ylabel('Amplitude (arb. units)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/rasmus/Documents/ThesisWork/Thesistex/DigitalSetup/signals.png', format='png')
plt.show()
