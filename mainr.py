import numpy as np
import cv2
from globalmatting import *
from guidedfilter import *
import os

#for x in range(trimap.shape[1]):
#	for y in range(trimap.shape[0]):
#		if trimap[y][x]==255:
#			count=count+1

name= "GT04-image.png"
#pth="./test/images/" + name
img = cv2.imread("GT04-image.png", cv2.IMREAD_COLOR)
trimap = cv2.imread("GT04-trimap.png", cv2.IMREAD_GRAYSCALE)
print trimap 
#alpha= np.zeros(trimap.shape[0], trimap.shape[1])

foreground= np.zeros(img.shape)

alpha = np.zeros((trimap.shape[0], trimap.shape[1]))

globalmatting(img, trimap, foreground, alpha) 

alpha= guided_filter(img, alpha, 10, 1e-5) 
for x in range(trimap.shape[1]):
	for y in range(trimap.shape[0]):
		if trimap[y][x]==0:
			alpha[y][x]=0
		elif trimap[y][x] == 255:
			alpha[y][x]= 255 

final = "matted" + name

cv2.imwrite(final, alpha)


