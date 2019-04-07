# coding: utf-8
import uproot as ur
import numpy as np
def getBinCenters(bins):
    """ calculate center values for given bins """
    return np.array([np.mean([bins[i],bins[i+1]]) for i in range(0,
len(bins)-1)])

def getLiveTime(file_name):
    """ retrieves experiment's live time in percent. necessary
xtrg/latch histogram is generated during preanalysis (or cooking) of
data. """
    tfile = ur.open(file_name)
    # get histogram storing live times (one entry per scaler event);
    #tested with uproot 3.2.12
    hvals, hbins = tfile['scalers/hLiveTime_Proj_Vme_Xtr'].numpy()
    # calculate mean of values
    mean = np.average(getBinCenters(hbins), weights=hvals)
    return mean
getLiveTime('data/finalData/Data1793_cooked.root')
