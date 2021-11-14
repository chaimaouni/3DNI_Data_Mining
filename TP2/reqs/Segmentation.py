#!/usr/bin/env python
# coding: utf-8

# In[221]:


import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import skimage.io
import matplotlib.pyplot as plt
import pandas as pd

# Loading image
image = cv2.imread('samples/eleph.jpg')
(h1, w1) = image.shape[:2]

# Change color to RGB (from BGR) 
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#reshape the image to a 2D array of pixels and 3 color values (RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = KMeans(n_clusters = 2)

labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

#reshape the feature vectors to images
quant = quant.reshape((h1, w1, 3))
image = image.reshape((h1, w1, 3))

# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

cv2.imwrite('samples/eleph2.jpg', quant)
image0 = cv2.imread('samples/eleph2.jpg')
plt.imshow(image0)

cv2.imread('samples/eleph.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[222]:


plt.imshow(image)


# In[199]:


get_ipython().system('pip install opencv-python')


# In[223]:


import skimage.io
import matplotlib.pyplot as plt

img_path = "samples/eleph.jpg"
img = skimage.io.imread(img_path)/255.0

def plotnoise(img, mode, r, c, i):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg)
    else:
        plt.imshow(img)
    plt.title(mode)
    plt.axis("off")

plt.figure(figsize=(18,24))
r=4
c=2
plotnoise(img, "gaussian", r,c,1)
plotnoise(img, "localvar", r,c,2)
plotnoise(img, "poisson", r,c,3)
plotnoise(img, "salt", r,c,4)
plotnoise(img, "pepper", r,c,5)
plotnoise(img, "s&p", r,c,6)
plotnoise(img, "speckle", r,c,7)
plotnoise(img, None, r,c,8)
plt.show()


# In[224]:


from scipy.cluster.vq import kmeans, vq
import seaborn as sns


# In[225]:


# Loading image
image2 = cv2.imread('samples/landscape.jpeg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
plt.imshow(image2)


# In[226]:


r = []
g = []
b = []
 
for row in image2:
    for pixel in row:
        # A pixel contains RGB values
        r.append(pixel[0])
        g.append(pixel[1])
        b.append(pixel[2])
 
df = pd.DataFrame({'red':r, 'green':g, 'blue':b})
df.head()


# In[227]:


distortions = []
num_clusters = range(1, 7)
 
# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(df[['red','green','blue']].values.astype(float), i)
    distortions.append(distortion)
 
# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})
 
# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()


# In[228]:


# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image2.reshape((-1, 3))

# convert to float
pixel_values = np.float32(pixel_values)


# In[229]:


# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image2.reshape((-1, 3))

# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)


# In[230]:


def CulsterImg(k):
    # number of clusters (K)
    #k = 5
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image2.shape)
    return segmented_image


# In[231]:


K_2 = CulsterImg(2)
K_4 = CulsterImg(4)
K_6 = CulsterImg(6)


# In[232]:


# show the image
plt.figure(figsize=(15,24))

plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(image2)

plt.subplot(2,2,2)
plt.title('Image Clustered with k=2')
plt.imshow(K_2)

plt.subplot(2,2,3)
plt.title('Image Clustered with k=4')
plt.imshow(K_4)

plt.subplot(2,2,4)
plt.title('Image Clustered with k=6')
plt.imshow(K_6)


plt.show()


# In[234]:


def segmentNoizyImage(noizyImg):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = noizyImg.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)
    
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = noizyImg.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(noizyImg.shape)
    
    # show the image
    plt.imshow(segmented_image)
    plt.show()


# In[192]:


img2 = skimage.io.imread('samples/landscape_2.jpg')/255.0
gimg = skimage.util.random_noise(img2, mode='salt')

#plt.figure(figsize=(18,24))
#plt.imshow(gimg)
#plt.show()


# In[193]:


#segmentNoizyImage(gimg)

