import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#function to generate LoG kernel
def generate_LoG(sigma):
    #Adjust kernel size
    n = math.ceil(sigma*6)
    #x and y range to center the kernel
    x = np.array(range(-n//2,n//2 +1))
    x=np.reshape(x,(1,x.shape[0]))
    y = np.array(range(-n//2,n//2 +1))
    y=np.reshape(y,(y.shape[0],1))    

    x_filter = np.exp(-((x**2)/(2.*(sigma**2))))
    y_filter = np.exp(-((y**2)/(2.*(sigma**2))))
    #LoG kernel
    LoG_filter = (-(2*(sigma**2)) + (x**2 + y**2) ) * (x_filter*y_filter) * (1/(2*(np.pi*sigma**4)))
    return LoG_filter

#Function for convolution with appropriate padding
def conv2d(image,kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros(image.shape)
    
    px = (kernel.shape[0]-1)
    py = (kernel.shape[0]-1)
    pad = np.zeros((image.shape[0] + px, image.shape[1] + py))
    #zero padding
    pad[int(px/2):int(image.shape[0]+px/2),int(py/2):int(image.shape[1]+py/2)] = image[:,:]
    #Convolution
    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            output[x, y]=(kernel * pad[x: x+kernel.shape[0], y: y+kernel.shape[1]]).sum()         
    return output

#Non Max Suppression
def non_max_supp(Laplacian_space, Layers):
    #Size of images
    r, c = Laplacian_space[0].shape
    
    #Suppression in individual layer
    supp_each_layer = np.zeros((Layers,r, c), dtype='float32') 
    for i in range(Layers):
        py_layer = Laplacian_space[i, :, :]
        for j in range(r):
            for k in range(c):
                supp_each_layer[i, j, k] = np.max(py_layer[j:j+3 , k:k+3])
                
    #Suppression among different layers
    supp_layer_lvl = np.zeros((Layers, r, c), dtype='float32')
    #Start from 2nd bottom most layer
    for i in range(1, np.size(supp_each_layer,1)-1):
            for j in range(1, np.size(supp_each_layer,2)-1):
                supp_layer_lvl[:, i, j] = np.max(supp_each_layer[:, i-1:i+2 , j-1:j+2])
    
    #select the max in every layer and image
    max_space = np.multiply((supp_layer_lvl == supp_each_layer), supp_layer_lvl)
    return max_space

#Detecting the maxima and thresholding
def thresholding(max_space, scale, sigma,th=0.03):
    blobs = []                                        

    for j in range(max_space[0].shape[0]):
        for i in range(max_space[0].shape[1]):
            local_neigh = max_space[:, j:j+3 , i:i+3]            
            maxima = np.max(local_neigh)                     
            if maxima >= th:                            
                layer,y,x = np.unravel_index(local_neigh.argmax(), local_neigh.shape)
                scaled_sd = np.power(scale,layer)*sigma
                blobs.append((scaled_sd,j+y, i+x))                    
    return blobs                                       

#Input image and normalise it
img = cv2.imread("wolves.png", 0) / 255.0

#Standard deviation for gaussian
sigma = 2

#Scaling Constant
scale = (2**0.5)

#Number of layers in the image pyramid
Layers = 9

#Building the laplacian scale space
Img_conv = []
#Run for n iterations
for i in range(Layers):
    #Scaling the standard deviation in each iteration
    #Producing the LoG kernel for required standard deviation
    LoG_filter = generate_LoG(sigma * np.power(scale, i))
    #Convolve the LoG filter
    filtered_LoG = conv2d(img, LoG_filter)
    #Squaring the LoG response
    squared_LoG = np.square(filtered_LoG)
    Img_conv.append(squared_LoG)

Laplacian_space = np.array(Img_conv)
#Non Max Suppression
max_space = non_max_supp(Laplacian_space, Layers)

#detecting the blobs
blobs =  thresholding(max_space, scale, sigma,th=0.03)
#remove duplicates
blobs = set(blobs)
#Plotting the circles on the image
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
for blob in blobs:
    scaled_sd,y,x = blob
    ax.add_patch(plt.Circle((x, y), scaled_sd*scale, color='red', linewidth=0.3, fill=False))
        
ax.plot()  
plt.show()