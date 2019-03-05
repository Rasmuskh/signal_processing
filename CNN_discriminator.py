# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd

np.random.seed(666)
#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.callbacks import ModelCheckpoint

waveFormLegth=300

#training set
Training_set = dd.read_parquet('data/2019-02-13/T.pq', engine='pyarrow').query('amplitude>20').reset_index()
neutrons = Training_set.query('55000<tof<78000 and 1000*%d<cfd_trig_rise<1000*(1204-%d)'%(waveFormLegth, waveFormLegth))
neutrons = neutrons.compute()
neutrons = neutrons.drop('index', axis=1)
neutrons = neutrons.reset_index()
gammas = Training_set.query('20000<tof<35000 and 1000*%d<cfd_trig_rise<1000*(1204-%d)'%(waveFormLegth, waveFormLegth))
gammas = gammas.compute()
gammas = gammas.drop('index', axis=1)
gammas = gammas.reset_index()

#testset
df_test = dd.read_parquet('data/2019-02-13/V.pq', engine='pyarrow').query('amplitude>20').reset_index()
#df_test = dd.read_parquet('data/2019-02-13/test/test2min.parquet', engine='pyarrow').reset_index()
df_test = df_test.query(' 1000*%d<cfd_trig_rise<1000*(1204-%d)'%(waveFormLegth, waveFormLegth))
df_test = df_test.compute()
df_test = df_test.drop('index', axis=1)
df_test = df_test.reset_index()


def get_samples(df):
    S = np.array([None]*df.shape[0])
    #S = [0]*len(df)
    for i in range(0, len(df)):
        S[i] = df.samples[i][int(0.5 + df.cfd_trig_rise[i]/1000)-20: int(0.5 + df.cfd_trig_rise[i]/1000)+waveFormLegth-20]
    return S

Sn = get_samples(neutrons)
Sy = get_samples(gammas)
St = get_samples(df_test)


L=min([len(neutrons), len(gammas)])
print(L, ' samples of each species will be used')
window_width = len(Sn[0])#n_train.window_width[0]
r = 0.8
X1=np.stack(Sn[0:int(r*L)])#n_train.samples)
X2=np.stack(Sy[0:L])#y_train.samples)
x_train = np.concatenate([X1, X2]).reshape(L+int(r*L),window_width,1)
y_train = np.array([1]*int(r*L) + [0]*L)

x_test=np.stack(St)#df_test.samples)
x_test=x_test.reshape(len(x_test), window_width, 1)

#model definition
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(window_width, 1)))
model.add(Dropout(0.08))
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(filters=16, kernel_size=5, strides=1, activation='relu'))
model.add(Dropout(0.08))
model.add(MaxPooling1D(2, stride=2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid', name='preds'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=56, epochs=5)#, validation_split=0.1)
predictions = model.predict(x_test)
df_test['pred']=predictions
df_test0 = df_test.query('pred<0.5')
df_test1 = df_test.query('pred>=0.5')

#ToF spectrum
plt.hist(df_test.tof/1000, bins=250, range=(0,500), alpha=0.25, label='Sum')
plt.hist(df_test0.tof/1000, bins=250, range=(0,500), histtype='step', lw=1.5, label='Gamma')
plt.hist(df_test1.tof/1000, bins=250, range=(0,500), histtype='step', lw=1.5, label='Neutron')
plt.legend()
plt.title('ToF spectrum \nfiltered by convolutional neural network\nTrained on 45 minute dataset, here tested on 15 minute dataset')
plt.ylabel('Counts')
plt.xlabel('t(ns)')
plt.show()

#Prediction space
plt.hist(df_test.pred, bins=50, label='Neutron region = 0.5-1, gamma region=0-0.5')
plt.legend()
plt.title('CNN prediction space\n the final layers output is the logistic function, so it is bounded between 0 and 1')
plt.ylabel('Counts')
plt.xlabel('CNN prediction')
plt.show()

#prediction vs QDC
plt.scatter(df_test0.qdc_lg/100, df_test0.pred, alpha=0.45, label='Gamma')
plt.scatter(df_test1.qdc_lg/100, df_test1.pred, alpha=0.45, label='Neutron')
plt.xlim(0,12500)
plt.legend()
plt.title('CNN predictions versus longgate QDC values')
plt.xlabel('qdc channel')
plt.ylabel('CNN prediction')
plt.show()
T=Training_set.query('0<tof<100000 and 0<cfd_trig_rise<650000')

dummy=df_test.query('-0.4<=ps<0.6')
H = sns.JointGrid(dummy.ps, dummy.pred)
H = H.plot_joint(plt.hexbin, cmap='inferno', gridsize=(120,120))
H.ax_joint.set_xlabel('Tail/total')
H.ax_joint.set_ylabel('CNN prediction')
_ = H.ax_marg_x.hist(dummy.ps, color="purple", alpha=.5, bins=np.arange(-0.4, 0.6, 0.01))
_ = H.ax_marg_y.hist(dummy.pred, color="purple", alpha=.5, orientation="horizontal", bins=np.arange(0, 1, 0.01))
plt.setp(H.ax_marg_x.get_yticklabels(), visible=True)
plt.setp(H.ax_marg_y.get_xticklabels(), visible=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # shrink fig so cbar is visible
cbar_ax = H.fig.add_axes([0.92, 0.08, .02, 0.7])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
plt.show()





# T=Training_set.query('0<tof<100000 and 0<cfd_trig_rise<650000')
# plt.hexbin(T.cfd_trig_rise.compute(), T.tof.compute(), gridsize=100)
# plt.xlabel('cfd_trigger time within window (ps)')
# plt.ylabel('tof (ps)')
# plt.show()

