
#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set(style='ticks')
import pandas as pd
import numpy as np
#import dask.dataframe as dd
from scipy.signal import convolve



waveFormLegth=300

#training set
neutrons = pd.read_parquet('data/finalData/CNN/neutrons.pq', engine='pyarrow').query('55000<tof<75000 and amplitude<610').reset_index()
gammas = pd.read_parquet('data/finalData/CNN/gammas1.pq', engine='pyarrow').query('22000<tof<30000 and amplitude<610').reset_index()


L=min(len(gammas), len(neutrons))
neutrons=neutrons.head(L)
gammas=gammas.head(L)

#testset
df_test = pd.read_parquet('data/finalData/CNN/test.pq', engine='pyarrow').reset_index()#.query('(20000<=tof<=30000) or (53000<=tof<=78000)').reset_index()
df_test10min = pd.read_parquet('data/finalData/CNN/test10min.pq', engine='pyarrow').reset_index()


# def get_samples_train(df):
#     S = np.array([None]*df.shape[0])
#     #S = [0]*len(df)
#     for i in range(0, len(df)):
#         p = 50
#         r = np.random.randint(0,p)
#         randArr1= np.random.randint(-2,2,r)
#         randArr2= np.random.randint(-2,2,p-r)
#         S[i] = df.samples[i][int(0.5 + df.cfd_trig_rise[i]/1000)-20: int(0.5 + df.cfd_trig_rise[i]/1000)+waveFormLegth-p-20].astype(np.float64)#/df.amplitude[i]
#         R = list(randArr1) + list(S[i])+list(randArr2)
#         S[i] = np.array(R)
#     return S
def get_samples(df):
    S = np.array([None]*df.shape[0])
    #S = [0]*len(df)
    for i in range(0, len(df)):
        S[i] = df.samples[i][int(0.5 + df.cfd_trig_rise[i]/1000)-20: int(0.5 + df.cfd_trig_rise[i]/1000)+waveFormLegth-20].astype(np.float64)#/df.amplitude[i]
    return S

Sn = get_samples(neutrons)
Sy = get_samples(gammas)
#release memory
neutrons=0;gammas=0
St = get_samples(df_test)
St10 = get_samples(df_test10min)

window_width = len(Sn[0])#n_train.window_width[0]
r = 1
X1=np.stack(Sn[0:int(r*L)])#n_train.samples)
X2=np.stack(Sy[0:L])#y_train.samples)
x_train = np.concatenate([X1, X2]).reshape(L+int(r*L),window_width,1)
y_train = np.array([1]*int(r*L) + [0]*L)

x_test=np.stack(St)#df_test.samples)
x_test=x_test.reshape(len(x_test), window_width, 1)
y_test = df_test.y

x_test10=np.stack(St10)#df_test.samples)
x_test10=x_test10.reshape(len(x_test10), window_width, 1)

#model definition
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=9, strides=4, activation='relu', input_shape=(window_width, 1), padding='same'))
#keras.layers.BatchNormalization()
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(filters=16, kernel_size=5, strides=2, activation='relu', padding='same'))
#keras.layers.BatchNormalization()
model.add(MaxPooling1D(2, stride=2))

model.add(Flatten(name='flat'))
model.add(Dense(1, activation='sigmoid', name='preds'))

opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
filepath="CNN_models3/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
# Fit the model
epochs = 200
hist = model.fit(x_train, y_train, batch_size=50, epochs=epochs, validation_data=(x_test, y_test),verbose=2, callbacks=callbacks_list)
print(hist.history)

#model_path='weights-improvement-08-0.87.hdf5'
#model=keras.models.load_model('CNN_models3/%s'%model_path)



#predictions = model.predict(x_test)
#df_test['pred'] = predictions
#df_test0 = df_test.query('pred<0.5')
#df_test1 = df_test.query('pred>=0.5')

predictions = model.predict(x_test10)
df_test = df_test10min
df_test['pred'] = predictions
df_test0 = df_test.query('pred<0.5')
df_test1 = df_test.query('pred>=0.5')

# int_layer_model = Model(inputs=model.input, outputs=model.get_layer('flat').output)
# params = int_layer_model.predict(x_test).reshape(3, len(df_test))
# df_test['params0']=params[0]
# df_test['params1']=params[1]
# df_test['params2']=params[2]



#train, validation
kernel=np.array([1,1,1,1,1,1,1,1,1])/9
plt.figure(figsize=(6.2,2.4))
plt.plot(hist.history['acc'], label='Training accuracy', color='blue')
plt.plot(range(4,epochs-4), convolve(hist.history['acc'], kernel, method='direct',mode='same')[4:-4], label='Accuracy averaged', color='green')
plt.plot(hist.history['val_acc'], label='Validation accuracy', color='red')
plt.plot(range(4, epochs-4), convolve(hist.history['val_acc'], kernel, method='direct', mode='same')[4:-4], label='Validation accuracy averaged', color='orange')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and validation accuracy')
plt.legend()
plt.tight_layout()
plt.show()


#ToF spectrum
plt.hist(df_test.tof/1000, bins=75, range=(-20,130), alpha=0.25, label='Sum')
plt.hist(df_test0.tof/1000, bins=75, range=(-20,130), histtype='step', lw=1.5, label='Gamma')
plt.hist(df_test1.tof/1000, bins=75, range=(-20,130), histtype='step', lw=1.5, label='Neutron')
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

# #prediction vs QDC
# plt.scatter(df_test0.qdc_lg/1000, df_test0.pred, alpha=0.45, label='Gamma')
# plt.scatter(df_test1.qdc_lg/1000, df_test1.pred, alpha=0.45, label='Neutron')
# plt.xlim(0,12500)
# plt.legend()
# plt.title('CNN predictions versus longgate QDC values')
# plt.xlabel('qdc channel')
# plt.ylabel('CNN prediction')
# plt.show()


# Ecal = np.load('data/finalData/Ecal_D.npy')/1000
# df_test['E'] = Ecal[1] + Ecal[0]*df_test['qdc_lg_fine']
# dummy = df_test.query('E<6')
# plt.hexbin(dummy.E, dummy.pred, norm=mc.LogNorm())
# plt.show()

# dummy=df_test.query('-0.4<=ps<1')
# H = sns.JointGrid(dummy.ps, dummy.pred)
# H = H.plot_joint(plt.hexbin, cmap='inferno', gridsize=(50,50), norm=mc.LogNorm())
# H.ax_joint.set_xlabel('Tail/total')
# H.ax_joint.set_ylabel('CNN prediction')
# _ = H.ax_marg_x.hist(dummy.ps, color="purple", alpha=.5, bins=np.arange(0, 1, 0.01))
# _ = H.ax_marg_y.hist(dummy.pred, color="purple", alpha=.5, orientation="horizontal", bins=np.arange(0, 1, 0.01))
# plt.setp(H.ax_marg_x.get_yticklabels(), visible=True)
# plt.setp(H.ax_marg_y.get_xticklabels(), visible=True)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # shrink fig so cbar is visible
# cbar_ax = H.fig.add_axes([0.92, 0.08, .02, 0.7])  # x, y, width, height
# plt.colorbar(cax=cbar_ax)
# plt.show()





# T=Training_set.query('0<tof<100000 and 0<cfd_trig_rise<650000')
# plt.hexbin(T.cfd_trig_rise.compute(), T.tof.compute(), gridsize=100)
# plt.xlabel('cfd_trigger time within window (ps)')
# plt.ylabel('tof (ps)')
# plt.show()


