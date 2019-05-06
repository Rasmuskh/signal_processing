# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.ticker as plticker
import seaborn as sns; sns.set(color_codes=True, style='ticks')
import sys
import dask.dataframe as dd
sys.path.append('../../analog_tof/')
sys.path.append('../tof')
import pyTagAnalysis as pta
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

#1.055+179/1000/2

d_detect = 1.055+0.005+179/1000/2
c=0.299792458 # m/ns

#Load in dataframes
#Load the digitized
#D = pd.read_parquet('../data/finalData/finalData.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 50<=amplitude<6100')
D = pd.read_parquet('../cnn_test1hour.pq', engine='pyarrow', columns=['cfd_trig_rise', 'window_width', 'channel', 'amplitude', 'qdc_lg', 'qdc_sg', 'ps', 'pred', 'tof', 'baseline_std']).query('channel==0 and 20<cfd_trig_rise/1000<window_width-500 and 50<=amplitude<6100')

fsg=4900
flg=250
D['ps'] = ((flg*500+D['qdc_lg'])-(fsg*60+D['qdc_sg']))/(flg*500+D['qdc_lg']).astype(np.float64)
Dcal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_D.npy')
#Tshift_D = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_D.npy')
D['E'] = (D.qdc_lg*Dcal[0]+Dcal[1])/1000
Tshift_D=25.59788102555002*1000
D['tof'] = (D['tof'] - Tshift_D)/1000 + d_detect/c

#Load the analog
A = pta.load_data('../data/finalData/Data1793_cooked.root')
A['qdc_lg'] = A['qdc_det0']
A['qdc_sg'] = A['qdc_sg_det0']
#A['amplitude'] = 1
flg=0
fsg=2
A['ps'] = ((flg*500+A.qdc_det0)-(fsg*60+A.qdc_sg_det0))/(flg*500+A.qdc_det0).astype(np.float64)
Acal=np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Ecal_A.npy')
#Tshift_A = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/Tshift_A.npy')
Tshift_A = 379.5970031276691
Tcal = np.load('/home/rasmus/Documents/ThesisWork/code/tof/data/finalData/T_cal_analog.npy')
A['E'] = A.qdc_lg*Acal[0]+Acal[1]
A['tof'] = 1000 - A['tdc_det0_yap0']
A['tof'] = (A['tof'] - Tshift_A)*(-Tcal[0]) + d_detect/c
cmap='viridis'

def getBinCenters(bins):
    """ From Hanno Perrey calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0,len(bins)-1)])
def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
def get_error(x0, sigma, dx0,dsigma, frac):
    return ( dx0**2 + (sigma**2)*4*(log(frac))**2  )**(1/2)

def get_err(df,mode, cut, Blim, Ylim, Nlim):
    if mode == 'CNN':
        N_B = len(df.query('pred>=%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
        Y_B = len(df.query('pred<%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
    elif mode == 'CC':
        N_B = len(df.query('ps>=%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
        Y_B = len(df.query('ps<%s and %s<tof<=%s'%(cut, Blim[0], Blim[1])))/(Blim[1]-Blim[0])
    print('%s: N_B=%s'%(mode, N_B))
    print('%s: Y_B=%s'%(mode, Y_B))
    #expected neutrons in gamma region
    N_Y_exp = N_B*(Ylim[1]-Ylim[0])
    #expected gammas in neutron region
    Y_N_exp = Y_B*(Nlim[1]-Nlim[0])
    if mode == 'CNN':
        # neutrons in gamma region
        N_Y = len(df.query('pred>=%s and %s<tof<=%s'%(cut, Ylim[0], Ylim[1])))
        # gammas in neutron region
        Y_N = len(df.query('pred<%s and %s<tof<=%s'%(cut, Nlim[0], Nlim[1])))
    elif mode =='CC':
        # neutrons in gamma region
        N_Y = len(df.query('ps>=%s and %s<tof<=%s'%(cut, Ylim[0], Ylim[1])))
        # gammas in neutron region
        Y_N = len(df.query('ps<%s and %s<tof<=%s'%(cut, Nlim[0], Nlim[1])))
    #err = diff from expectation
    N_Y_err = N_Y - N_Y_exp
    Y_N_err = Y_N - Y_N_exp
    #
    tot_Y = len(df.query('%s<tof<=%s'%Ylim))-N_Y_exp
    tot_N = len(df.query('%s<tof<=%s'%Nlim))-Y_N_exp
    print('False Neutron rate = ',100*N_Y_err/tot_Y,'% of gammas')
    print('False gamma rate = ',100*Y_N_err/tot_N,'% of neutrons')
    return (N_Y_err/(tot_Y), Y_N_err/(tot_N), N_B/(N_B+Y_B))


def tof_hist(df, outpath, qdc_min, mode, window, bins, fontsize, txt):
    dummy=df.query('%s<qdc_lg'%(qdc_min))

    plt.figure(figsize=(6.2,4))
    plt.hist(dummy.tof, bins, range=window, histtype='step', lw=1.5)
    plt.axvline(x=0, linestyle='--', color='black')
    plt.xlabel('ToF (ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    plt.xlim(-35,130)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    textstr='%s'%txt
    plt.text(80, 230, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='none', edgecolor='blue', pad=0.5, boxstyle='square'))
    plt.text(7, 460, "Neutrons", verticalalignment='top', fontsize=10)
    plt.text(-33, 600, "Gamma-rays", verticalalignment='top', fontsize=10)

    ax.annotate('', xy=(37, 100), xytext=(16 ,400), fontsize= fontsize,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),)
    ax.annotate(' ', xy=(5, 350), xytext=(-25,560), fontsize= fontsize,
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),)

    Elow = 1.5
    Ehigh = 7
    c=299792458
    m=939.565#*c
    x=d_detect
    tnlow = int(0.5 + (1/2*m*x**2/(Ehigh*c**2))**(1/2)*10**9)
    print(tnlow)
    tnhigh = int(0.5 + (1/2*m*x**2/(Elow*c**2))**(1/2)*10**9)
    print(tnhigh)
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    print(len(dummy))
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    dummy['Eneutron'] = E
    

    plt.hist(dummy.tof, bins=tnhigh-tnlow, range=(tnlow, tnhigh), color='red', alpha=0.7)
    with sns.axes_style("ticks"):
        plt.axes([.5, .5, .45, .4], facecolor='white')


    S=pd.read_csv('scherzingerData.csv', names=['x','y'])
    S.x -= (S.x[1]-S.x[0])/2
    if mode == 'A':
        plt.hist(E, range=(Elow,Ehigh), bins=int(0.5+(Ehigh-Elow)*2), color='red', label='Neutrons', alpha=0.7)
        scale=16000#1.6*max(np.histogram(E, range=(Elow,Ehigh), bins=int(0.5+(Ehigh-Elow)*2.4))[0]) / max(S.y[int(len(S.y)/2):len(S.y)])
        plt.step(S.x, S.y*scale, label='Scherzinger' )
        plt.ylim(0,1.4*plt.gca().get_ylim()[1])
    else:
        plt.hist(E, range=(Elow,Ehigh), bins=int(0.5+(Ehigh-Elow)*5), color='red', label='$\mathrm{E_{neutron}}$', alpha=0.7)
        scale=5100#max(np.histogram(E, range=(Elow,Ehigh), bins=int(0.5+(Ehigh-Elow)*3))[0]) / max(S.y)
        plt.step(S.x, S.y*scale, label='Scherzinger' )
        plt.ylim(0,1.4*plt.gca().get_ylim()[1])

    plt.ylabel('Counts', fontsize= fontsize)
    plt.xlabel('E_{neutron} (MeV)', fontsize= fontsize)
    plt.legend(fontsize= fontsize)
    loc = plticker.MultipleLocator(base=1.0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(loc)

    plt.tight_layout()
    plt.savefig(outpath+'tof.pdf', format='pdf')
    plt.show()

def Edep_Eneutron(df, outpath, fontsize, title, mode):

    Elow = 2
    Ehigh = 7
    c=299792458
    m=939.565#*c
    x=d_detect
    tnlow = (1/2*m*x**2/(Ehigh*c**2))**(1/2)*10**9
    print(tnlow)
    tnhigh = (1/2*m*x**2/(Elow*c**2))**(1/2)*10**9
    print(tnhigh)
    dummy=df.query('%s<tof<%s'%(tnlow, tnhigh)).reset_index()
    print(len(dummy))
    #plt.hist(dummy.tof, range=(0,100), bins=100)
    #plt.show()
    v=(x/(dummy.tof.astype(np.float64)*10**(-9)))
    vnat=v/c
    E=1/2*m*(vnat)**2
    dummy['Eneutron'] = E

    fontsize = 10
    plt.figure(figsize=(6.2, 6.2))
    left=-0.2
    right=1.1
    bins=30
    d=dummy
    colorList = ['red', 'blue', 'orange']
    thrList = [50, 103, 155]
    f = 100


    # plt.subplot(2,2,1)
    # for i in range(0, len(thrList)):
    #     plt.hist(d.query('amplitude>%s'%thrList[i]).E/d.query('amplitude>%s'%thrList[i]).Eneutron, range=(left, right), bins=bins, histtype='step', lw=1.5, color=colorList[i], label='thr=%s mV'%int(0.5+(thrList[i]/1.024)))
    # plt.plot(x, gaus(x, popt[0], popt[1], popt[2]), color=colorList[0], label='Gaussian fit')
    # plt.axvline(x=popt[1], color='black', ls='--', lw=1, label='$\mathrm{x_0}$=%s'%round(popt[1],3))

    # plt.xlabel('$ \dfrac{\mathrm{E_{deposition}}}{E_{neutron}}$', fontsize=fontsize)
    # plt.ylabel('Counts')
    # plt.legend(loc=2)
    # plt.xlim(left, right)
    # ax1=plt.gca()
    # ax1.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    # plt.tight_layout()






    #..........
    plt.subplot(3,1,2)
    plt.scatter(d.Eneutron, d.E, s=50, alpha=0.08, label='', zorder=0)
    plt.xlim(0,6)
    elim = (Elow, 5.2)#(Elow, 4.7)
    plt.ylim(0, 3)
    L=13
    x0List = [0]*(L-1)
    EnList = np.linspace(elim[0],elim[1],L)
    Enlist_centers = Enlist_centers = (EnList+ (EnList[1]- EnList[0])/2)[0:-1]
    errorList=[0]*len(x0List)
    plt.axvline(x=EnList[0]+(L-1)*(EnList[1]-EnList[0]), linestyle='--', color='black', lw=1)
    k=0
    for i in range (0,len(x0List)):
        plt.axvline(x=EnList[i], linestyle='--', color='black', lw=1)
        H = np.histogram(d.query('(%s)<Eneutron<(%s)'%(EnList[i], EnList[i+1])).E, range=(0, 6), bins=60)
        x = getBinCenters(H[1])
        p, c = curve_fit(gaus, x, H[0], p0=[max(H[0]), np.mean(x), 10])
        errorList[i] = p[2]
        x0List[i] = p[1]
        plt.subplot(3,1,1)

        if ((i==0) or (i==5) or (i==10)):
            plt.subplot(3,1,1)
            plt.plot(x, gaus(x, p[0], p[1], p[2]), label='slice %s'%i, color=colorList[k])
            plt.step(x=x, y=H[0], color=colorList[k])
            k+=1
            plt.legend()
            plt.xlabel('$\mathrm{E_{dep}}$ ($\mathrm{MeV}_{ee}$)')
            plt.xlim(0,3)
            plt.ylabel('Counts')
        plt.subplot(3,1,2)
    #plt.subplot(1,1,1)
    #plt.legend()

    plt.subplot(3,1,2)
    plt.errorbar(Enlist_centers, x0List, yerr=errorList, marker='o', color='red',fmt='o', zorder=2)
    plt.gca().set_xticklabels([''])
    plt.ylabel('$\mathrm{E_{dep}}$ ($\mathrm{MeV}_{ee}$)')
    def conv(En, par1, par0=0):
        return par0 + par1*(0.83*En - 2.82*(1 - exp(-0.25*(En)**(0.93) ) ) )
        #return par0 + -par1*2.82*(1 - exp(-0.25*(En)**(0.93)) +0.83*En )

    #Not using offset
    popt_Curve,pcov = curve_fit(conv, xdata=Enlist_centers, ydata=x0List, p0=[1], bounds=((0),(5)))
    print(popt_Curve)
    x = np.linspace(0,10,100)
    plt.plot(x, conv(x, popt_Curve[0]), label='Fit: C=%s'%round(popt_Curve[0],2), color='orange', zorder=1)


    plt.legend(fontsize=fontsize)
    ax3=plt.gca()
    ax3.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)

    H = np.histogram(d.E/conv(d.Eneutron, popt_Curve), range=(left, right), bins=bins*f)
    x = getBinCenters(H[1])
    popt,pcov = curve_fit(gaus, x, H[0], p0=[max(H[0]), np.mean(x), 10])
    popt[0] *= f

    #panel 2
    plt.subplot(3,1,3)
    plt.scatter(d.Eneutron,d.E/conv(d.Eneutron, popt_Curve),  s=50, alpha=0.08)
    plt.xlim(0, 6)
    plt.ylim(-0.5, 2)
    L=13
    x0List = [0]*(L-1)
    EnList = np.linspace(Elow,5.2,L)
    Enlist_centers = Enlist_centers = (EnList+ (EnList[1]- EnList[0])/2)[0:-1]
    errorList=[0]*len(x0List)
    plt.axvline(x=EnList[0]+(L-1)*(EnList[1]-EnList[0]), linestyle='--', color='black', lw=1)
    for i in range (0,len(x0List)):
        plt.axvline(x=EnList[i], linestyle='--', color='black', lw=1)
        H = np.histogram(d.query('(%s)<Eneutron<(%s)'%(EnList[i], EnList[i+1])).E/conv(d.Eneutron, popt_Curve), range=(-1,5), bins=600)
        x = getBinCenters(H[1])
        p, c = curve_fit(gaus, x, H[0], p0=[max(H[0]), np.mean(x), 10])
        errorList[i] = p[2]
        x0List[i] = p[1]


        #plt.subplot(1,1,1)
        #plt.plot(x, gaus(x, p[0], p[1], p[2]))
        #plt.step(x=x, y=H[0], label='%s'%i)
        #plt.legend()
        #plt.show()

    plt.errorbar(Enlist_centers, x0List, yerr=errorList, marker='o', color='red',fmt='o')
    plt.axhline(y=popt[1], color='black', ls='--', lw=1)#, label='$\mathrm{x_0}$=%s'%round(popt[1],3))
    print('zzzzzzzzzzz', popt[1])
    plt.legend()
    plt.xlabel('$\mathrm{E_{neutron}}$ ($\mathrm{MeV}$)')
    plt.ylabel('$\mathrm{ \dfrac{E_{dep}\;[MeV_\mathit{ee}]}{E_{neutron}\;[MeV_\mathit{ee}]}}$', fontsize=fontsize)


    plt.tight_layout()
    plt.savefig(outpath+'N_E.pdf', format='pdf')
    plt.show()


def tof_hist_filt(df, outpath, qdc_min, cut, mode, psmode, CNN, window, bins, fontsize, Blim, Ylim, Nlim):
    dummy=df.query('%s<qdc_lg and -20<tof<=150'%(qdc_min))
    logon=False
    plt.figure(figsize=(6.2,2.8))
    plt.hist(dummy.tof, bins, range=window,histtype='step', lw=1, label='All', log=logon)
    if CNN==True:
        plt.hist(dummy.query('pred<%s'%cut).tof, bins, range=window, alpha=0.35, label='Gamma-rays', color='blue', log=logon)
        plt.hist(dummy.query('pred>=%s'%cut).tof, bins, range=window, alpha=0.35, label='Neutrons', color='red', log=logon)
        outpath+='CNN'
    else:
        plt.hist(dummy.query('ps<%s'%cut).tof, bins, range=window, alpha=0.35, label='Gamma-rays', color='blue', log=logon)
        plt.hist(dummy.query('ps>=%s'%cut).tof, bins, range=window, alpha=0.35, label='Neutrons', color='red', log=logon)
    plt.xlabel('ToF (ns)', fontsize= fontsize)
    plt.ylabel('Counts', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.ylim(0,1200)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()

    err_Y, err_N, N_to_Y_rati_background = get_err(dummy, psmode, cut, Blim, Ylim, Nlim)


    textstr='%s'%(mode)

    plt.text(17, 1100, textstr, fontsize= fontsize, verticalalignment='top',bbox=dict(facecolor='white', edgecolor='blue', pad=0.5, boxstyle='square'))
    # ax.annotate('$\gamma-n\; region$', xy=(45, 230), xytext=(55 ,600), fontsize= fontsize,
    #         arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    # ax.annotate('$\gamma-\gamma\; region$', xy=(5, 600), xytext=(16,800), fontsize= fontsize,
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    # ax.annotate('$Background\; region$', xy=(100, 150), xytext=(110,500), fontsize= fontsize,
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2, frac=0.10, headwidth=9),
    # )
    plt.axvline(x=Nlim[0], ymin=0, ymax=0.2, color='black', ls='-', lw=1)
    plt.axvline(x=Nlim[1], ymin=0, ymax=0.2, color='black', ls='-', lw=1, label='$\gamma n$ region: %s%% mislabeled '%int(round(100*err_N,0)))
    plt.axvline(x=Ylim[0], ymin=0, ymax=0.2, color='black', ls='--', lw=1)
    plt.axvline(x=Ylim[1], ymin=0, ymax=0.2, color='black', ls='--', lw=1, label='$\gamma \gamma$ region: %s%% mislabeled'%int(round(100*err_Y,0)))
    plt.axvline(x=Blim[0], ymin=0, ymax=0.2, color='black', ls=':', lw=1)
    plt.axvline(x=Blim[1], ymin=0, ymax=0.2, color='black', ls=':', lw=1, label='Background region:\n%s%% neutrons'%int(round(100*N_to_Y_rati_background,0)))
    plt.legend()
    plt.savefig(outpath+'ToF_filt.pdf', format='pdf')
    plt.show()

def psd(df, outpath, CNN, down, cut, up, qdc_min, title, arrow1, arrow2, box, fontsize):
    plt.figure(figsize=(6.2,3.1))
    if CNN==True:
        dummy=df.query('%s<pred<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.pred, gridsize=(100, 100), cmap=cmap, extent=(0,7, down, up), vmin=0, vmax=1000)
        plt.ylabel('CNN prediction', fontsize= fontsize)
        outpath+='CNN'
    else:
        dummy=df.query('%s<ps<%s and E<6 and %s<qdc_lg'%(down, up, qdc_min))
        plt.hexbin( dummy.E, dummy.ps, gridsize=(100, 100), cmap=cmap, extent=(0,7, down, up))
        plt.ylabel('PS', fontsize= fontsize)
    plt.xlabel('E ($\mathrm{MeV}$$_{ee})$', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    ax = plt.gca()
    textstr='%s'%title
    plt.text(box[0], box[1], textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    plt.text(5,cut+0.10,'Neutrons', fontsize= fontsize, color='white', verticalalignment='top')
    plt.text(5,cut-0.05,'Gamma-rays', fontsize= fontsize, color='white', verticalalignment='top')
    ax.annotate('2.23 $\mathrm{MeV}$', xy=arrow1[0:2], xytext=arrow1[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('4.44 $\mathrm{MeV}$', xy=arrow2[0:2], xytext=arrow2[2:4], color='white', fontsize= fontsize,
            arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    plt.colorbar()
    ax.tick_params(axis = 'both', which = 'both', labelsize =  fontsize)
    plt.tight_layout()
    plt.savefig(outpath+'psd.png', format='png')
    plt.show()

def tof_E(df, outpath, fontsize, title):
    plt.figure(figsize=(6.2, 2.8))
    plt.hexbin(df.tof, df.E, extent=(0,100, 0, 6), norm=mc.LogNorm(), cmap='viridis')
    plt.xlabel('Time of flight (ns)', fontsize= fontsize)
    plt.ylabel('E ($\mathrm{MeV}_{ee}$)', fontsize= fontsize)
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
    plt.xlabel('ToF (ns)', fontsize= fontsize)
    #plt.title(title, fontsize=12)
    plt.axhline(y=cut, linestyle='--', color='white', lw=1)
    plt.colorbar()
    ax = plt.gca()
    textstr='%s'%title
    plt.text(30, psdown+0.07, textstr, fontsize= fontsize, color='white', verticalalignment='top',bbox=dict(facecolor='None', edgecolor='white', pad=0.5, boxstyle='square'))
    ax.annotate('Neutrons', xy=arrow_xy_neutron, xytext=txt_xy_neutron, color='white', fontsize= fontsize,
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, frac=0.10, headwidth=9),
    )
    ax.annotate('Gamma-rays', xy=arrow_xy_gamma, xytext=txt_xy_gamma, color='white', fontsize= fontsize,
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
CNN_cut = 0.5
#tof_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', window=(-20,130), bins=150, qdc_min=0, fontsize=fontsize, txt="ToF spectrum\nDigital setup", mode='D')
#tof_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', window=(-20,130), bins=150, qdc_min=500, fontsize=fontsize, txt="ToF spectrum\nAnalog setup", mode='A')

# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', CNN=False, cut=DigitalCut, down=0, up=0.5, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2, 0.15, 2.5, 0.01], arrow2=[4.2, 0.14, 4.7, 0.01], box=[2, 0.48])
# psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/',  CNN=True, cut=CNN_cut, down=0, up=1, qdc_min=0, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2.1, 0.1, 2.5, 0.25], arrow2=[4.2, 0.05, 4.7, 0.25], box=[2, 0.65])
# psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/',  CNN=False, cut=AnalogCut, down=0, up=0.5, qdc_min=500, fontsize=fontsize, title="--- Discrimination cut", arrow1=[2, 0.2, 2.5, 0.11], arrow2=[4.2, 0.2, 4.7, 0.11], box=[2, 0.48])

#tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=-0.1, psup=0.5, tofdown=0, tofup=100, qdc_min=0, cut=DigitalCut, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[5, 0.4], txt_xy_neutron=[70, 0.45], arrow_xy_gamma=[3.5, 0.25], arrow_xy_neutron=[50, 0.3], CNN=False)
#tof_psd(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=0, cut=CNN_cut, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[10, 0.4], txt_xy_neutron=[60, 0.7], arrow_xy_gamma=[4, 0.2], arrow_xy_neutron=[45, 0.9], CNN=True)
#tof_psd(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', psdown=0, psup=1, tofdown=0, tofup=100, qdc_min=500, cut=AnalogCut, fontsize=fontsize, title="--- Discrimination cut", txt_xy_gamma=[10, 0.48], txt_xy_neutron=[60, 0.48], arrow_xy_gamma=[4, 0.28], arrow_xy_neutron=[45, 0.38], CNN=False)

# Cgamma=d_detect/c
# Ylim = (Cgamma-5,Cgamma+5)
# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=DigitalCut, window=(-20,160), bins=180, qdc_min=0, fontsize=fontsize, psmode="CC", mode='Digital setup\nCC', CNN=False, Blim=(70, 150), Nlim = (29, 62), Ylim = Ylim)
# tof_hist_filt(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', cut=CNN_cut, window=(-20,160), bins=180, qdc_min=0, fontsize=fontsize, psmode="CNN", mode='Digital setup\nCNN', CNN=True, Blim=(70, 150), Nlim = (29, 62), Ylim = Ylim)
# tof_hist_filt(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', cut=AnalogCut, window=(-20,160), bins=180, qdc_min=500, fontsize=fontsize, psmode="CC", mode='Analog setup\nCC', CNN=False,  Blim=(70, 150), Nlim = (29, 62), Ylim = Ylim)

# compare_PSD(D)


#tof_E(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Digital setup')
#tof_E(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup')

Edep_Eneutron(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', fontsize=12, title='Analog setup', mode='D')
#Edep_Eneutron(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', fontsize=12, title='Analog setup', mode='A')

# qdc_hist(D, '/home/rasmus/Documents/ThesisWork/Thesistex/DigitalResults/', bins=160, window=(0,16), fontsize=fontsize, title="Energy deposition spectrum\nAnalog setup")
# qdc_hist(A, '/home/rasmus/Documents/ThesisWork/Thesistex/AnalogResults/', bins=80, window=(0,8), fontsize=fontsize, title="Energy deposition spectrum\nDigital setup")
