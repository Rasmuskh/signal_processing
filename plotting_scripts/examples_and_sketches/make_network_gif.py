# coding: utf-8
import imageio
images=[]
for i in range(0, 150):
    images.append(imageio.imread('NetworkViz_Gif/frame%d.png'%i))

imageio.mimsave('CNN.gif', images, duration=[0.1]*(len(images)))
