# coding: utf-8
import imageio
images=[]
for i in range(0, 101,5):
    images.append(imageio.imread('hex/psd%d.png'%i))
for i in range(0,15):
    images.append(imageio.imread('hex/psd_cut.png'))

imageio.mimsave('psd.gif', images, duration=[0.1]*(len(images)))
