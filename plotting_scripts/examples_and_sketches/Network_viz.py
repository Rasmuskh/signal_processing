# coding: utf-8
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='ticks')
import numpy as np
from scipy.signal import convolve
from scipy import asarray as ar,exp
import dask.dataframe as dd
import keras
from keras.models import Model


D=dd.read_parquet('../../data/finalData/finalData.pq/',engine='pyarrow').query('channel==0 and baseline_std<1').head(50000)
k=5
N=D.query('pred>0.95 and 0<amplitude<1500').reset_index().head(k)
Y=D.query('pred<0.05 and 0<amplitude<1500').reset_index().head(k)
S=D.query('0.495<pred<0.505 and 0<amplitude<1500').reset_index().head(k)
D=0
pre=20
post=280

def view_types():
    for i in range(0, len(N)):
        sN=int(N.cfd_trig_rise[i]/1000+0.5)
        sY=int(Y.cfd_trig_rise[i]/1000+0.5)
        sS=int(S.cfd_trig_rise[i]/1000+0.5)
        plt.plot((N.samples[i][sN-pre:sN+post]+N.fine_baseline_offset[i]/1000), color='red', alpha=0.5)
        plt.plot( (Y.samples[i][sY-pre:sY+post]+Y.fine_baseline_offset[i]/1000), color='blue', alpha=0.5)
        #plt.plot((S.samples[i][sS-pre:sS+post]+S.fine_baseline_offset[i]/1000)/, color='green', alpha=0.5)
    plt.legend(['Neutron', 'Gamma-ray'])
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    for i in range(0, len(N)):
        sN=int(N.cfd_trig_rise[i]/1000+0.5)
        sY=int(Y.cfd_trig_rise[i]/1000+0.5)
        sS=int(S.cfd_trig_rise[i]/1000+0.5)
        plt.plot((N.samples[i][sN-pre:sN+post]+N.fine_baseline_offset[i]/1000)/N.amplitude[i], color='red', alpha=0.5)
        plt.plot( (Y.samples[i][sY-pre:sY+post]+Y.fine_baseline_offset[i]/1000)/Y.amplitude[i], color='blue', alpha=0.5)
        #plt.plot((S.samples[i][sS-pre:sS+post]+S.fine_baseline_offset[i]/1000)/S.amplitude[i], color='green',alpha=0.5)
    plt.legend(['Neutron', 'Gamma-ray'])
    plt.xlabel('Time (ns)')
    plt.ylabel('Normalized Amplitude')
    plt.tight_layout()
    plt.show()


model_path='../../CNN_models3/model_zero_53.hdf5'
model=keras.models.load_model('%s'%model_path)
int_layer_model = Model(inputs=model.input, outputs=model.get_layer('flat').output)

#int_layer_model._make_predict_function()
pre_trig=20
CNN_window = 300


def occlude(df, txt, c):
    occl_pred = [0]*len(df)
    #lin_pred =  [0]*len(N)
    for i in range(0, 1):
        #lin_pred[i] = np.array([0]*CNN_window, dtype=np.float64)
        occl_pred[i] = np.array([0]*CNN_window, dtype=np.float64)
        for u in range(0, CNN_window):
            s = int(0.5 + df.cfd_trig_rise[i]/1000)
            wave_raw = np.array(df.samples[i][s-pre_trig:s+CNN_window-pre_trig])
            wave = np.array(df.samples[i][s-pre_trig:s+CNN_window-pre_trig])
            x = np.ones(300)
            x[u-16:u+15]=0
            wave=wave*x
            #wave[u] = 0
            wave_raw = wave_raw.reshape(1, CNN_window, 1).astype(np.float64)
            wave = wave.reshape(1, CNN_window, 1).astype(np.float64)
            #lin_pred[i][u] = int_layer_model.predict(wave_raw)[0][0].sum()
            #occl_pred[i][u] = int_layer_model.predict(wave)[0][0].sum()
            occl_pred[i][u] = model.predict(wave)[0][0]
            #N['lin_pred'] = lin_pred
            df['occl'] = occl_pred

    for i in range(0, len(N)):
        sdf=int(df.cfd_trig_rise[i]/1000 + 0.5)
        plt.plot((df.samples[i][sdf-pre_trig:sdf+CNN_window-pre_trig]+df.fine_baseline_offset[i]/1000)/df.amplitude[i], color=c, alpha=0.75, label='%s'%txt)
        plt.ylim(-1.25,0.25)
        plt.xlabel('Time (ns)')
        plt.ylabel('Normalized amplitude')
        plt.legend(loc=3)
        plt.gca().twinx()
        plt.plot(df.occl[i], color='orange', label='Masked pulse prediction')
        plt.axhline(y=df.pred[i], color='black', linestyle='--', label='Original prediction')
        plt.ylim(-0.25,1.25)
        plt.xlim(0,150)
        plt.ylabel('Prediction')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()

def occlude2(df, txt, c):
    occl_pred = [0]*len(df)
    #lin_pred =  [0]*len(N)
    for i in range(0, 1):
        #lin_pred[i] = np.array([0]*CNN_window, dtype=np.float64)
        occl_pred[i] = np.array([0]*CNN_window, dtype=np.float64)
        for u in range(0, CNN_window):
            s = int(0.5 + df.cfd_trig_rise[i]/1000)
            wave_raw = np.array(df.samples[i][s-pre_trig:s+CNN_window-pre_trig])
            wave = np.array(df.samples[i][s-pre_trig:s+CNN_window-pre_trig])
            x = np.ones(300)
            x[u-16:u+15]=0
            wave=wave*x
            #wave[u] = 0
            wave_raw = wave_raw.reshape(1, CNN_window, 1).astype(np.float64)
            wave = wave.reshape(1, CNN_window, 1).astype(np.float64)
            #lin_pred[i][u] = int_layer_model.predict(wave_raw)[0][0].sum()
            #occl_pred[i][u] = int_layer_model.predict(wave)[0][0].sum()
            occl_pred[i][u] = model.predict(wave)[0][0]
            #N['lin_pred'] = lin_pred
            df['occl'] = occl_pred
            wave_raw = wave_raw.reshape(CNN_window).astype(np.float64)
            wave = wave.reshape(CNN_window).astype(np.float64)
            if (u%1==0) and (u<151):
                plt.plot(wave_raw/df.amplitude[i], label='original pulse', color='green')
                plt.plot(wave/df.amplitude[i], label='masked pulse', color='orange')
                plt.ylim(-1.25,0.25)
                plt.xlim(0,150)
                plt.text(100, -0.5, 'Prediction=%s'%round(occl_pred[i][u], 2), bbox=dict(boxstyle='square', facecolor='white'))
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Normalized amplitude')
                plt.tight_layout()
                plt.savefig('NetworkViz_Gif/frame%s'%u)
                plt.close()

#occlude(N, txt='Neutron pulse', c='red')
#occlude(Y, txt='Gamma-ray pulse', c='blue')
occlude(S, txt='Ambiguous pulse', c='green')
#view_types()
