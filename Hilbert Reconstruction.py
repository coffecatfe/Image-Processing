
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from math import pi
from numpy.fft import fft2,fftshift,ifftshift,ifft2


# In[ ]:


def mat2gray(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x));
    return x


# In[ ]:


A0 = cv2.imread(r'C:\Users\user\Desktop\T=12_0.bmp',cv2.IMREAD_GRAYSCALE)

A180 = cv2.imread(r'C:\Users\user\Desktop\T=12_180.bmp',cv2.IMREAD_GRAYSCALE)


# In[ ]:


A0 = A0.astype(float)
A180 = A180.astype(float)

[p,p] = np.shape(A0)
mask = np.zeros([p,p],dtype=complex)


for x in range(p):
    for y in range(p):
        u = x-p/2+1
        v = y-p/2+1
        try:
            mask[x,y] = complex(u,v)/np.sqrt(u**2+v**2)
        except ZeroDivisionError:
            mask[x,y] = 0


A = A0-A180
A_ft = fftshift(fft2(A))
#h = mat2gray(abs(A_ft))
#cv2.imshow('FFT',h)

B_ft = A_ft*mask
B = ifft2(ifftshift(B_ft))
B = abs(B)

k = A+B*complex(0,1)
k1 = abs(k)
h = mat2gray(k1)*255
cv2.imwrite('hilbert.bmp',h)


