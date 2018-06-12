import reader as rdr
import cfd
import pandas as pd
import time
import numpy as np


def pulse2dataframe(filename):
    tstart=time.time()
    frame=rdr.load_events(filename)
    nTimeResets=0
    tstamp=[0]*len(frame)
    refpoint=[0]*len(frame)
    noevent=[False]*len(frame)
    lzcross=[0]*len(frame)
    rzcross=[0]*len(frame)
    for n in range(0,len(frame)):
        if n%1000==0:
            print('Event', n, '/', len(frame))
        if n>0:
            if frame.TimeStamp[n]<frame.TimeStamp[n-1]:
                nTimeResets+=1
        tstamp[n]=frame.TimeStamp[n]+nTimeResets*2147483647#+refpoint[n]

        if np.count_nonzero(frame.Samples[n]) == 0:
            noevent[n] = True
            continue
        else:
            refpoint[n] = cfd.shifter(frame.Samples[n])
            for u in range(refpoint[n]-1,-1,-1):
                if frame.Samples[n][u]<1:
                    lzcross[n]=u
                    break
            for y in range(refpoint[n]+1,len(frame.Samples[n])):
                if frame.Samples[n][y]<1:
                    rzcross[n]=y
                    break
  

    Frame=pd.DataFrame({'TimeStamp': tstamp,
                        'Samples' : frame.Samples,
                        'Baseline' : frame.Baseline,
                        'refpoint':refpoint,
                        'Left':lzcross,
                        'Right': rzcross,
                        'noevent': noevent})
    tstop=time.time()
    print('processing time = ',tstop-tstart)
    return Frame

