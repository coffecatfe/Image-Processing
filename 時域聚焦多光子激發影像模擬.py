

import cv2
import numpy as np
from numpy.fft import fft2,fftshift,ifftshift,ifft2
from math import pi
from cv2 import imread,imshow


# In[3]:


#基本參數
NA = 1.2
ran = 0.25
depth = 6
z = np.arange(0,depth+ran,ran)
k = int(depth/ran+1)


# In[4]:


def mat2gray(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x));
    return x

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def optFW(L):
    w0 = L/(pi*NA)
    z0 = pi*(w0**2)/L
    w = w0*np.sqrt(1+(z/z0)**2)
    w = w/0.132  #DMD1pixel對應ＣＣＤ的pixel數
    FWHM = w*np.sqrt(2*np.log10(2))
    return FWHM


# In[5]:


A = cv2.imread(r'C:\Users\user\Desktop/USAF.bmp',cv2.IMREAD_GRAYSCALE)
#A = cv2.imread('C:/Users/A21C96410/Desktop/sim/USAF.bmp',cv2.IMREAD_GRAYSCALE)
#cv2.imshow('USAF',A)


# In[6]:


[W,L]=np.shape(A)
origin=A.copy()
[p,p] = np.shape(A)  #旋轉後擷取的邊長
D = np.zeros((p,p))
x=y=np.arange(p)

[X,Y] = np.meshgrid(x,y)
D = np.sqrt((X-np.round(p/2)-1)**2+(Y-np.round(p/2)-1)**2)


# In[7]:


#模擬照明時的MTF
Li = 0.8
FWHM = optFW(Li)

PSFi = np.zeros((k,p,p),dtype=float)
ci = np.zeros(FWHM.shape).astype(float)
OTFi=np.zeros(np.shape(PSFi),dtype=complex)
MTFi=np.zeros(np.shape(PSFi),dtype=float)

#估算PSF的參數
for num in range(k):
    ci[num] = FWHM[num]/(2*np.sqrt(2*np.log10(2)))
    PSFi[num,:] = np.exp(-(D**2)/(2*(ci[num]**2)))
for num in range(k):
    OTFi[num,:,:]=fftshift(fft2(PSFi[num,:,:]**0.5))
    MTFi[num,:,:]=mat2gray(np.abs(OTFi[num,:,:]))


# In[8]:


#模擬收光時的MTF
Le = 0.514
FWHM = optFW(Le)

ce = np.zeros(FWHM.shape).astype(float)
PSFe = np.zeros((k,p,p),dtype=float)
OTFe=np.zeros(np.shape(PSFe),dtype=complex)
MTFe=np.zeros(np.shape(PSFe),dtype=float)


for num in range(k):
    ce[num] = FWHM[num]/(2*np.sqrt(2*np.log10(2)))
    PSFe[num,:] = np.exp(-(D**2)/(2*(ce[num]**2)))
for num in range(k):
    OTFe[num,:,:]=fftshift(fft2(PSFe[num,:,:]**0.5))
    MTFe[num,:,:]=mat2gray(np.abs(OTFe[num,:,:]))


# In[9]:


#強度系數
FWHM = 3
zr = FWHM/3.46
I = (1+(z/zr)**2)**(-0.5)
Iw = 2*sum(I)-I[0]


# In[10]:


origin_u = origin.copy().astype(float)


# In[11]:


u_ft = np.zeros((k,p,p),dtype=complex)
u = np.zeros((k,p,p),dtype=float)

for num in range(k):
    u_ft[num,:,:] = fftshift(fft2(origin_u))
    u_ft[num,:,:] = u_ft[num,:,:]*MTFe[num,:,:]
    u[num,:,:] = abs(ifft2(ifftshift(u_ft[num,:,:])))




# In[12]:


#每層影像乘以各自的權重後加總

#WF每層各自能量總和
swf = u.sum(axis=1).sum(axis=1)
u[num,:,:] = u[num,:,:]*(1/swf[num])*(I[num]/Iw)

pic_u = 2*u.sum(axis=0)-u[0,:,:]
h = mat2gray(pic_u)*255
cv2.imwrite('raw_data.bmp', h)


# In[13]:


#條紋週期
T = 12
Tt = T/2 
#Tv = T/2
#Th = T/2


# In[14]:


#設計pattern/同調
#垂直條紋

#WF單層能量總和
patternt0 = np.zeros((p,p),dtype=float)
kt = 2*pi/Tt

#設定相位
degree = 180
phase = degree*pi/180

for x in range(p):
    for y in range(p):
        patternt0[x,y]=(1+np.cos(kt*(x-y)+phase))/2

Ut = patternt0**(0.5);
t_ft = fftshift(fft2(Ut))

It =  np.zeros((k,p,p),dtype=float)
for num in range(k):
    It[num,:,:] = abs(ifft2(ifftshift(t_ft*MTFi[num,:,:]))**2);


patternt = It**2;


# In[38]:


#設定pattern軸向解析度
FWHM = 3
zr = FWHM/3.46
Ip = (1+(z/zr)**2)**(-0.5)


# In[49]:


#斜條紋

origin_n = origin.copy().astype(float)
n = n_t = np.zeros((k,p,p),float)
n_ft = np.zeros((k,p,p),complex)

for num in range(k):
    n[num,:,:] = origin_n*patternt[num,:,:]
    n_ft[num,:,: ] = fftshift(fft2(n[num,:,:]))*MTFe[num,:,:]
    n_t[num,:,:] = abs(ifft2(ifftshift(n_ft[num,:,:])))



# for num=1:k
# n_v(:,:,num)=mat2gray(n_v(:,:,num)).*Ip(:,num);
# h=n_v(:,:,num);
# figure(num);imshow(h);
# M(num) = getframe;

# movie2avi(M,'C:\Users\APL\Desktop\debug\raw_data\20v.avi','FPS',2);

 
sn = n_t.sum(axis=1).sum(axis=1)
for num in range(k):
    n_t[num,:,:] = n_t[num,:,:]*(swf[1]/sn[num])*Ip[num]


pic_t = n_t.sum(axis=0)
h = mat2gray(pic_t)*255
#imshow(h)

cv2.imwrite('T'+str(T)+'_'+str(degree)+'.bmp', h)
