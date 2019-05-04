# coding: utf-8
import numpy as np
from numpy.random import rand
from math import cos, pi

L=1000000
N=np.array([0]*L)
for i in range(0, L):
    E=1
    n=0
    while(E>0.302):
        n+=1
        r=rand()*pi
        E=cos(r)**2
    N[i]=n
print(N.mean())
