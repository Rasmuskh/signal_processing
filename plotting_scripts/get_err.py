# coding: utf-8
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.ticker as plticker
import seaborn as sns; sns.set(color_codes=True, style='whitegrid')
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta

c=0.299792458 # m/ns

#Load in dataframes
#Load the digitized
D = pd.read_parquet('../data/finalData/finalData.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 50<=amplitude<6100')
fsg=4900
flg=250
D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_D.npy')
Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
D['tof'] = (D['tof'] - Tshift_D[1])/1000 + 1.055/c

#Load the analog
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['qdc_lg'] = A['qdc_det0']
A['qdc_sg'] = A['qdc_sg_det0']
flg=0
fsg=2
A['ps'] = ((flg*500+A.qdc_det0)-(fsg*60+A.qdc_sg_det0))/(flg*500+A.qdc_det0).astype(np.float64)
Acal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_A.npy')
Tshift_A = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_A.npy')
Tcal = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/T_cal_analog.npy')
A['E'] = A.qdc_lg*Acal[0]+Acal[1]
A['tof'] = 1000 - A['tdc_det0_yap0']
A['tof'] = (A['tof'] - Tshift_A[1])*(-Tcal[0]) + 1.055/c

cmap='viridis'
def get_err(df,mode, cut, Blim, Ylim, Nlim):
    if mode == 'CNN':
        N_B = len(df.query('pred>=%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
        Y_B = len(df.query('pred<%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
    elif mode == 'CC':
        N_B = len(df.query('ps>=%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
        Y_B = len(df.query('ps<%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
    print('%s: N_B=%s'%(mode, N_B))
    print('%s: Y_B=%s'%(mode, Y_B))
    N_Y_exp = N_B*(Ylim[1]-Ylim[0]) 
    Y_N_exp = Y_B*(Nlim[1]-Nlim[0])
    if mode == 'CNN':
        N_Y = len(df.query('pred>=%s and %s<tof<=%s'%(cut, Ylim[0], Ylim[1])))
        Y_N = len(df.query('pred<%s and %s<tof<=%s'%(cut, Nlim[0], Nlim[1])))
    elif mode =='CC':
        N_Y = len(df.query('ps>=%s and %s<tof<=%s'%(cut, Ylim[0], Ylim[1])))
        Y_N = len(df.query('ps<%s and %s<tof<=%s'%(cut, Nlim[0], Nlim[1])))
    N_Y_err = N_Y - N_Y_exp
    Y_N_err = Y_N - Y_N_exp
    tot_Y = len(df.query('%s<tof<=%s'%Ylim))
    tot_N = len(df.query('%s<tof<=%s'%Nlim))
    print('False Neutron rate = ',100*N_Y_err/tot_Y,'% of gammas')
    print('False gamma rate = ',100*Y_N_err/tot_N,'% of neutrons')
    return (N_Y_err/tot_Y, Y_N_err/tot_N)

Adummy=A.query('E>0.8')
Ddummy=D#.query('E>1.5')
#plt.hist(D.tof, bins=800, range=(-50,150), alpha=0.5)
#plt.hist(Adummy.tof, bins=800, range=(-50,150), alpha=0.5)
#plt.show()
get_err(Ddummy, 'CNN', cut=0.5, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
get_err(Ddummy, 'CC', cut=0.222, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
get_err(Adummy, 'CC', cut=0.259,  Blim = (70, 150), Nlim = (29, 55), Ylim = (0,8))

D_CNNerrorlist=[0]*100
D_CCerrorlist=[0]*100
A_CCerrorlist=[0]*100
for i in range(0,100):
    ERR = get_err(Ddummy, 'CNN', cut=0.01*i, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
    #ERR = (ERR[0]**2+ERR[1]**2)**(1/2)
    ERR = (ERR[0]+ERR[1])
    D_CNNerrorlist[i] =ERR
    ERR = get_err(Ddummy, 'CC', cut=0.01*i, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
    #ERR = (ERR[0]**2+ERR[1]**2)**(1/2)
    ERR = (ERR[0]+ERR[1])
    D_CCerrorlist[i] =ERR
    ERR = get_err(Adummy, 'CC', cut=0.01*i,  Blim = (70, 150), Nlim = (29, 55), Ylim = (0,8)) 
    #ERR = (ERR[0]**2+ERR[1]**2)**(1/2)
    ERR = (ERR[0]+ERR[1])
    A_CCerrorlist[i] =ERR
    print(i)

CNN_Cut = np.argmin(D_CNNerrorlist)/100
DigitalCut = np.argmin(D_CCerrorlist)/100
AnalogCut = np.argmin(A_CCerrorlist)/100
# get_err(Ddummy, 'CNN', cut=CNN_Cut, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
# get_err(Ddummy, 'CC', cut=DigitalCut, Blim = (70, 150), Nlim = (32, 60), Ylim = (0,8))
# get_err(Adummy, 'CC', cut=AnalogCut,  Blim = (70, 150), Nlim = (29, 55), Ylim = (0,8))
# plt.plot(D_CNNerrorlist, label='Digital: CNN')
# plt.plot(D_CCerrorlist, label='Digital: CC')
# plt.plot(A_CCerrorlist, label='Analog: CC')
# plt.legend()
# plt.show()

# CNN_Cut = np.argmin(D_CNNerrorlist)/100
# DigitalCut = np.argmin(D_CCerrorlist)/100
# AnalogCut = np.argmin(A_CCerrorlist)/100
# fontsize=10
# plt.subplot(2,1,1)
# plt.hist(D.tof, bins=200, range=(-50,150), histtype='step')
# plt.hist(D.query('pred>=0.5').tof, bins=200, range=(-50,150), alpha=0.5)
# plt.hist(D.query('pred<0.5').tof, bins=200, range=(-50,150), alpha=0.5)
# plt.subplot(2,1,2)
# plt.hist(D.tof, bins=200, range=(-50,150), histtype='step')
# plt.hist(D.query('pred>=%s'%CNN_Cut).tof, bins=200, range=(-50,150), alpha=0.5)
# plt.hist(D.query('pred<%s'%CNN_Cut).tof, bins=200, range=(-50,150), alpha=0.5)
# plt.show()
