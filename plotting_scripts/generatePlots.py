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
D = pd.read_parquet('../data/finalData/finalData.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 40<=amplitude<6100')
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
    return (N_Y_err/tot_Y, Y_N_err/tot_N, N_B/(N_B+Y_B))

def tof_hist(df, outpath, tNmax, qdc_min, mode, window, bins, fontsize, tnlow, tnhigh):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,3.1))
    plt.hist(dummy.tof, bins, range=window, histtype='step', lw=1.5)
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    textstr='%s'%mode
    plt.text(80, 280, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=(35, 230), xytext=(20 ,600), fontsize= fontsize,
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=(5, 850), xytext=(20,700), fontsize= fontsize,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.axvline(x=tnlow, ymin=0, ymax=0.22, color='red', ls='--', lw=1)
    plt.axvline(x=tnhigh, ymin=0, ymax=0.22, color='red', ls='--', lw=1)
    with sns.axes_style("ticks"):
        plt.axes([.52, .62, .45, .3], facecolor='white')
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    c=299792458
    m=939.565
    x=1.055
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    vlow = (x/(tnhigh*10**(-9)))/c
    vhigh = (x/(tnlow*10**(-9)))/c
    Elow = 1/2*m*(vlow)**2
    Ehigh = 1/2*m*(vhigh)**2
    sns.set_style("ticks")
    plt.hist(E, range=(Elow,Ehigh), bins=tnhigh-tnlow, histtype='step', color='red', label='Neutron\nenergy', lw=1.5)
    plt.ylabel('Counts', fontsize= fontsize)
    plt.xlabel('E(MeV)', fontsize= fontsize)
    plt.legend(fontsize= fontsize)
    loc = plticker.MultipleLocator(base=1.0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()
    plt.savefig(outpath+'tof.pdf', format='pdf')
    sns.set(color_codes=True, style='whitegrid')
    plt.show()

def Edep_Eneutron(df, outpath, fontsize, title, tnlow, tnhigh):
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    c=299792458
    m=939.565#*c
    x=1.055
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    vlow = (x/(tnhigh*10**(-9)))/c
    vhigh = (x/(tnlow*10**(-9)))/c
    Elow = 1/2*m*(vlow)**2
    Ehigh = 1/2*m*(vhigh)**2
    dummy['Eneutron'] = E

    plt.figure(figsize=(6.2,2.8))
    plt.hexbin(dummy.Eneutron, dummy.E, extent=(Elow, Elow+5, 0, 3), cmap='viridis', gridsize=(50,30))#, norm=mc.LogNorm())
    #plt.scatter(dummy.Eneutron, dummy.E, c=dummy.pred, cmap='viridis')#, extent=(Elow, Elow+5, 0, 3), cmap='viridis', gridsize=(50,30))#, norm=mc.LogNorm())
    plt.xlabel('Neutron energy $MeV$', fontsize= fontsize)
    plt.ylabel('E($MeV_{ee})$', fontsize= fontsize)
    ax = plt.gca()
    #textstr='%s'%title
    #plt.text(10, 5.5, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_Edep_Eneutron.pdf', format='pdf')
    plt.show()

def tof_hist_filt(df, outpath, qdc_min, cut, mode, psmode, CNN, window, bins, fontsize, Blim, Ylim, Nlim):
    dummy=df.query('%s<qdc_lg'%(qdc_min))
    plt.figure(figsize=(6.2,2.8))
    plt.hist(dummy.tof, bins, range=window,histtype='step', lw=1, label='All')
    if CNN==True:
        plt.hist(dummy.query('pred<%s'%cut).tof, bins, range=window, alpha=0.35, label='Gammas', color='blue')
        plt.hist(dummy.query('pred>=%s'%cut).tof, bins, range=window, alpha=0.35, label='Neutrons', color='red')
        outpath+='CNN'
    else:
        plt.hist(dummy.query('ps<%s'%cut).tof, bins, range=window, alpha=0.35, label='Gammas', color='blue')
        plt.hist(dummy.query('ps>=%s'%cut).tof, bins, range=window, alpha=0.35, label='Neutrons', color='red')
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()

    err_Y, err_N, N_to_Y_rati_background = get_err(dummy, psmode, cut, Blim, Ylim, Nlim)


    textstr='%s'%(mode)

    plt.text(20, 1100, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    # ax.annotate('$\gamma-n\; region$', xy=(45, 230), xytext=(55 ,600), fontsize= fontsize,
    #         arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    # ax.annotate('$\gamma-\gamma\; region$', xy=(5, 600), xytext=(16,800), fontsize= fontsize,
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    # ax.annotate('$Background\; region$', xy=(100, 150), xytext=(110,500), fontsize= fontsize,
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    plt.axvline(x=Nlim[0], ymin=0, ymax=0.25, color='black', ls='-', lw=1)
    plt.axvline(x=Nlim[1], ymin=0, ymax=0.25, color='black', ls='-', lw=1, label='$\gamma-n$ region: %s%% error'%round(100*err_N,2))
    plt.axvline(x=Ylim[0], ymin=0, ymax=0.25, color='black', ls='--', lw=1)
    plt.axvline(x=Ylim[1], ymin=0, ymax=0.25, color='black', ls='--', lw=1, label='$\gamma-\gamma$ region: %s%% error'%round(100*err_Y,2))
    plt.axvline(x=Blim[0], ymin=0, ymax=0.25, color='black', ls=':', lw=1)
    plt.axvline(x=Blim[1], ymin=0, ymax=0.25, color='black', ls=':', lw=1, label='Background region:\n%s%% neutrons'%round(100*N_to_Y_rati_background,2))
    plt.legend()
    plt.savefig(outpath+'ToF_filt.pdf', format='pdf')
    plt.show()

def psd(df, outpath, CNN, down, cut, up, qdc_min, title, arrow1, arrow2, box, fontsize):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.pred, gridsize=(100, 100), cmap=cmap, extent=(0,6, down, up))
        plt.ylabel('CNN prediction', fontsize= fontsize)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap=cmap, extent=(0,6, down, up))
        plt.ylabel('PS', fontsize= fontsize)
    plt.xlabel('E($MeV_{ee})$', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    ax = plt.gca()
    textstr='%s'%title
    plt.text(box[0], box[1], textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('2.23 $MeV$', xy=arrow1[0:2], xytext=arrow1[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('4.44 $MeV$', xy=arrow2[0:2], xytext=arrow2[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'psd.pdf', format='pdf')
    plt.show()

def tof_E(df, outpath, fontsize, title):
    plt.figure(figsize=(6.2, 2.8))
    plt.hexbin(df.tof, df.E, extent=(0,100, 0, 6), norm=mc.LogNorm(), cmap='viridis')
    plt.xlabel('Time of flight(ns)', fontsize= fontsize)
    plt.ylabel('E($MeV_{ee}$)', fontsize= fontsize)
    ax = plt.gca()
    #textstr='%s'%title
    #plt.text(10, 5.5, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Gammas', xy=(5,2), xytext=(10,3), color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Neutrons', xy=(45,2), xytext=(55,3), color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_E.pdf', format='pdf')
    plt.show()


def tof_psd(df, outpath, cut, psdown, psup, tofdown, tofup, qdc_min, title, txt_xy_neutron, txt_xy_gamma, arrow_xy_neutron, arrow_xy_gamma, CNN, fontsize):
    plt.figure(figsize=(6.2,2.8))
    if CNN==True:
        dummy=df.query('%s<pred<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof, dummy.pred, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('CNN prediction', fontsize= fontsize)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and %s<tof<%s and %s<qdc_lg'%(psdown, psup, tofdown, tofup, qdc_min))
        plt.hexbin( dummy.tof, dummy.ps, gridsize=(100, 100), cmap=cmap, norm=mc.LogNorm() )
        plt.ylabel('PS', fontsize= fontsize)
    plt.xlabel('ToF(ns)', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    plt.colorbar()
    ax = plt.gca()
    textstr='%s'%title
    plt.text(30, psdown+0.06, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=arrow_xy_neutron, xytext=txt_xy_neutron, color='white', fontsize= fontsize,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gammas', xy=arrow_xy_gamma, xytext=txt_xy_gamma, color='white', fontsize= fontsize,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'tof_psd.pdf', format='pdf')
    plt.show()

def compare_PSD(df):
    k=50000
    plt.scatter(df.head(k).E, df.head(k).ps, c=df.head(k).pred, cmap='viridis', alpha=1, s=3)
    plt.xlim(0,6)
    plt.ylim(-0.1, 0.5)
    plt.colorbar()
    plt.show()
    plt.hist(df.query('pred>=0.5').ps, range=(-0.1, 0.6), bins=100, alpha=0.5)
    plt.hist(df.query('pred<0.5').ps, range=(-0.1, 0.6), bins=100, alpha=0.5)
    plt.show()



def qdc_hist(df, outpath, bins, window, title, fontsize):
    plt.figure(figsize=(6.2,3.1))
    plt.hist(df.E, range=window, bins=500, log=True, histtype='step', lw=2)
    plt.xlabel('E($MeV_{ee}$)\n ', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    plt.title(title, fontsize= fontsize)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'qdc.pdf', format='pdf')
    plt.show()

fontsize = 10
DigitalCut = 0.222
AnalogCut = 0.259
# tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', window=(-20,130), bins=150, qdc_min=0, tNmax=75, fontsize=fontsize, mode="Time of flight spectrum\nDigital setup", tnlow=31, tnhigh=65)
# tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(-20,130), bins=150, qdc_min=500, tNmax=50, fontsize=fontsize, mode="Time of flight spectrum\nAnalog setup", tnlow=28, tnhigh=50)

# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', CNN=False, cut=DigitalCut, down=0, up=0.4, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2, 0.15, 2.5, 0.01], arrow2=[4.2, 0.14, 4.7, 0.01], box=[2, 0.38])
# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/',  CNN=True, cut=0.5, down=0, up=1, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2, 0.1, 2.5, 0.4], arrow2=[4.2, 0.05, 4.7, 0.4], box=[2, 0.65])
# psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/',  CNN=False, cut=AnalogCut, down=0.1, up=0.5, qdc_min=500, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2, 0.2, 2.5, 0.11], arrow2=[4.2, 0.2, 4.7, 0.11], box=[2, 0.48])

tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=-0.1, psup=0.5, tofdown=0, tofup=100, qdc_min=0, cut=DigitalCut, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[10, 0.3], txt_xy_neutron=[70, 0.45], arrow_xy_gamma=[5, 0.17], arrow_xy_neutron=[50, 0.3], CNN=False)
tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=0, cut=0.5, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[10, 0.4], txt_xy_neutron=[60, 0.7], arrow_xy_gamma=[4, 0.2], arrow_xy_neutron=[45, 0.9], CNN=True)
tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=500, cut=AnalogCut, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[10, 0.48], txt_xy_neutron=[60, 0.48], arrow_xy_gamma=[4, 0.28], arrow_xy_neutron=[45, 0.38], CNN=False)

# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=DigitalCut, window=(-20,160), bins=180, qdc_min=0, fontsize=fontsize, psmode="CC", mode='Digital setup\nCharge comparisson', CNN=False, Blim=(70, 150), Nlim = (32, 60), Ylim = (0,8))
# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=0.5, window=(-20,160), bins=180, qdc_min=0, fontsize=fontsize, psmode="CNN", mode='Digital setup\nCNN', CNN=True, Blim=(70, 150), Nlim = (32, 60), Ylim = (0,8))
# tof_hist_filt(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', cut=AnalogCut, window=(-20,160), bins=180, qdc_min=500, fontsize=fontsize, psmode="CC", mode='Analog setup\nCharge comparisson', CNN=False,  Blim=(70, 150), Nlim = (29, 55), Ylim = (0,8))

#compare_PSD(D)


tof_E(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Digital setup')
tof_E(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup')

Edep_Eneutron(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Analog setup', tnlow=30, tnhigh=65)
Edep_Eneutron(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup', tnlow=25, tnhigh=50)

# # qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=160, window=(0,16), fontsize=fontsize, title="Energy deposition spectrum\nAnalog setup")
# # qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=80, window=(0,8), fontsize=fontsize, title="Energy deposition spectrum\nDigital setup")
