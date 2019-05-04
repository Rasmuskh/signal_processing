# coding: utf-8

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

#train, validation
epochs=200
hist=np.load('cnn_training_dict.npy').item()
kernel=np.array([1,1,1,1,1,1,1,1,1])/9
x=np.linspace(1,200,200)
x2=np.linspace(8,200,192)
plt.figure(figsize=(6.2,2.6))
plt.plot(x,hist['acc'], label='Training accuracy', color='blue')
#plt.plot(x2, convolve(hist['acc'], kernel, method='direct',mode='same')[4:-4], label='Accuracy averaged', color='green')
plt.plot(x,hist['val_acc'], label='Validation accuracy', color='red')
#plt.plot(x2, convolve(hist['val_acc'], kernel, method='direct', mode='same')[4:-4], label='Validation accuracy averaged', color='orange')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'both', labelsize = 12)
plt.axvline(x=53, linestyle='--', color='green', label='Epoch = 53')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and validation accuracy')
plt.legend()
plt.tight_layout()
plt.show()
