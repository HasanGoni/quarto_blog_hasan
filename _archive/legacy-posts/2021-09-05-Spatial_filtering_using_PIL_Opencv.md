# Spatial Operation using Pillow and Opencv

Spatial operation uses pixels in neighborhood to determine the present pixel values. There are many applications of spatial transformations in computer vision

Here we will be working on 

* Linear Filtering

  * Filtering Noise
  * Gaussian Noise
  * Image Sharpening


* Edges
* Median

## Download Image

As usual we will download our required images


```python
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png   
```

    --2021-08-20 09:25:05--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7243 (7,1K) [image/jpeg]
    Saving to: 'cameraman.jpeg.1’
    
    cameraman.jpeg.1    100%[===================>]   7,07K  --.-KB/s    in 0s      
    
    2021-08-20 09:25:06 (1,92 GB/s) - 'cameraman.jpeg.1’ saved [7243/7243]
    
    --2021-08-20 09:25:06--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 473831 (463K) [image/png]
    Saving to: 'lenna.png.1’
    
    lenna.png.1         100%[===================>] 462,73K   639KB/s    in 0,7s    
    
    2021-08-20 09:25:08 (639 KB/s) - 'lenna.png.1’ saved [473831/473831]
    
    --2021-08-20 09:25:08--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 185727 (181K) [image/png]
    Saving to: 'barbara.png.1’
    
    barbara.png.1       100%[===================>] 181,37K   600KB/s    in 0,3s    
    
    2021-08-20 09:25:09 (600 KB/s) - 'barbara.png.1’ saved [185727/185727]
    
    


```python
import numpy as np
from PIL import Image
import cv2
from PIL import ImageOps
import matplotlib.pyplot as plt
```


```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

We will first copy our previously created functions


```python
def show_image(im,cmap=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im, cmap=cmap)
    ax.axis('off')



def side_by_side(im1, im2, 
                 im1_title='original',
                 im2_title=None, cmap1=None, cmap2=None):
    """Plot two images side  by side

    Args:
        im1 ([PIL:Opencv image]): [one image]
        im2 ([PIL:Opencv imag]): [another image]


    return None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    axes = ax.ravel()
    ax[0].imshow(im1, cmap=cmap1)
    ax[0].axis('off')
    ax[0].set_title(im1_title)
    ax[1].imshow(im2, cmap=cmap2)
    ax[1].axis('off')
    ax[1].set_title(im2_title,fontweight='bold')
```

### Linear Filtering:PIL

* Filtering actually helps to enhance the image

  * Enhancing could be removing the noise.
  * Or we sharpen a blurry image

* Convolution is a simple way to filter an image
* This filters are called kernels and there are different types of kernels, each kernel functions differently
* Convolution is very pretty common in modern deep learning applications.
* We simply take dot product of the kernel and an equally-sized portion of the image. We then shift the kernel and repeat the process.




```python
lena_pil = Image.open('lenna.png')
show_image(lena_pil)
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_12_0.svg)
    


So we will see how to remove noise  from an image. To remove noise, we need a noisy image. We will create a noisy image from this image


```python
rows, cols = lena_pil.size
noise = np.random.normal(0, 15, (rows, cols, 3)).astype(np.uint8)
noisy_image = lena_pil + noise
noisy_image = Image.fromarray(noisy_image)
side_by_side(im1=lena_pil, im2=noisy_image,
             im1_title='Original Image',
             im2_title='Noisy Image')


```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_14_0.svg)
    


### Filtering Noise

First we need to import `ImageFilter` to use the predefined filters

Smoothing filters average out the Pixels within neighbourhood, they are sometimes called `low pass filter`. For mean filtering kernel just averages out the kernels in a neighbourhood


```python
from PIL import ImageFilter 
```


```python
kernel = np.ones((5, 5))/36 # Creating 5 by 5 array where each value 1/36
kernel_filter = ImageFilter.Kernel((5, 5), kernel.flatten())
image_filtered = noisy_image.filter(kernel_filter)
side_by_side(im1=noisy_image, im2=image_filtered, im1_title='Noisy Image', im2_title='Filtered Image')
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_19_0.svg)
    


A smaller kernel make the image sharp, but filter less noise. Now we will use 3x3 kernel


```python
kernel = np.ones((3, 3))/36
kernel_filter = ImageFilter.Kernel((3, 3), kernel.flatten())
image_filtered_sm = noisy_image.filter(kernel_filter) 

```

We want to see side by side all 3 images


```python
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,10))
axes = ax.ravel()
axes[0].imshow(noisy_image)
axes[0].axis('off')
axes[0].set_title('Noisy Image')
axes[1].imshow(image_filtered)
axes[1].axis('off')
axes[1].set_title('kernel size 5x5')
axes[2].imshow(image_filtered_sm)
axes[2].axis('off')
axes[2].set_title('kernel size 3x3')

```




    Text(0.5, 1.0, 'kernel size 3x3')




    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_23_1.svg)
    


So we can see that `(5x5)` kernel removes more noise, but it is somehow blurry. However `(3x3)` kernel image more noisy, but it is somehow more sharp. Depending on the use case one needs to use the filter

### Gaussian Blurr

* So we have a predefined filter for `GaussianBlur` which is `ImageFilter.GaussianBlur()`
* Parameter is radius which is default 2. we will see if we change the defualt what happens


```python
# Filter image for GaussianBlur
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur)
side_by_side(im1=noisy_image,
             im2=image_filtered,
             im1_title='noiy image',
             im2_title='Guassian blurred image')
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_27_0.svg)
    


Let's change the default value to `4` 


```python
image_filter_4 = noisy_image.filter(ImageFilter.GaussianBlur(4))
```

Again I would like to plot 3 images side_by_side, what we have done. let's create another function, where we don't need to tell how much images, we need to plot


```python
def show_image_list(im_list,
                    title_list,
                    cmap_list = None):
    
    number_of_image = len(im_list)
    fig, ax = plt.subplots(nrows=1, ncols=number_of_image,
                           figsize=(18,10))
    axes = ax.ravel()
    for idx, i in enumerate(im_list):
        if cmap_list is not None:
            axes[idx].imshow(i, cmap=cmap_list[idx])
        else:
            axes[idx].imshow(i)
        axes[idx].axis('off')
        axes[idx].set_title(title_list[idx])


```


```python
show_image_list([noisy_image, image_filtered, image_filter_4], ['Noisy image', 'Guassian blur with 2 kernel', 'Guassian blur with 4 kernel'] )
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_32_0.svg)
    


So we have done blurring. Let's do the opposite, sharpening.

### Image Sharpening

* What image sharpening do is smooth the image, and calculate the derivatives. 
* We can do it using our kernel


```python
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]
                   ])
kernel = ImageFilter.Kernel((3, 3), kernel.flatten())            
sharp_image = lena_pil.filter(kernel)
show_image_list([lena_pil, sharp_image],['Original Image', 'Sharpened Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_36_0.svg)
    


Prevously we actually created out own filter. We can also use predefined filter


```python
sharpened_image = lena_pil.filter(ImageFilter.SHARPEN)
show_image_list([lena_pil, sharpened_image], ['Orginal Image', 'Sharp Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_38_0.svg)
    



```python
show_image_list([image_filtered, 
                 image_filtered.filter(ImageFilter.SHARPEN),
                 image_filter_4,
                 image_filter_4.filter(ImageFilter.SHARPEN) ],
['Guassian Blur\n kernel size 2', 'sharp after Guassian blur\n kernel 2', 'Guassian blur\n  kernel size 4','sharp after Guassian blur\n kernel size 4'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_39_0.svg)
    


### Linear Filtering:Opencv 

So in opencv we need to convert the image to rgb image, Now I am creating a function named as `oc` :opencv to convert that. Just don't want to write everytime the same function


```python
oc = lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
```


```python
lena_cv = cv2.imread('lenna.png')
show_image(oc(lena_cv))
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_43_0.svg)
    


So right now we will again create the noise and then an opencv noisy image


```python
lena_cv.shape
```




    (512, 512, 3)




```python
noise = np.random.normal(0, 15, (lena_cv.shape)).astype(np.uint8)
noisy_image = lena_cv + noise
show_image_list([oc(lena_cv), oc(noisy_image)],['Actual Image','Noisy Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_46_0.svg)
    


### Filtering Noise

like previously we will create our kernel first and then apply kernel to filter the noise


```python
kernel = np.ones((5, 5))/36
```

The function `filter2D` is similar to ImageFilter.kernel, here we have a `src`  which is the image, and kernel will be applied in each channel independently. Parameter `ddepth` is responsible for output size, here we will apply -1, as we want to have same input and output size.


```python
image_filter = cv2.filter2D(src=noisy_image,
                            ddepth=-1,
                            kernel=kernel)
show_image_list([oc(noisy_image), oc(image_filter)],['Original','filtered'])

```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_51_0.svg)
    


Last time we have seen the effect of bigger and smaller kernel size. Let's whether it is same for opencv also. Acutally it should be same, becuase opencv and PIL are some tools to do things, main mathematical function is same.

* Smaller kernel more sharp but remove less noise
* Bigger kernel are more blurry but remove more noise


```python
kernel1 = np.ones((7, 7)) / 36
kernel2 = np.ones((4, 4)) / 16
lena_kernel1 = cv2.filter2D(src=noisy_image,
                          ddepth=-1,
                          kernel=kernel1)
lena_kernel2 = cv2.filter2D(src=noisy_image,
                          ddepth=-1,
                          kernel=kernel2)
show_image_list([oc(noisy_image),
                 oc(image_filter),
                 oc(lena_kernel1),
                 oc(lena_kernel2) ],
                 ['Noisy Image', 'kernel size 5\n value $1/36$', 'kernel size 7\n value$1/36$','kernel 4\n with value $(1/16)$'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_54_0.svg)
    


We can see that kernel size 4 with value $1/16$ is sharp but more noise is there.

### Gaussian Blur

* The function `GaussianBlur` is good at removing noise but also try to save the edges.
* Parameters for GaussinaBlur
  * `src`: image and different number of channels will be processed independently
  * `ksize`: kernel size
  * `sigmaX`: kernel standard deviation in X direction
  * `sigmaY`: kernel standard deviation in Y direction


```python
# 4x4 kernel
image_filtered = cv2.GaussianBlur(noisy_image,
                                   (5, 5), sigmaX=4,
                                   sigmaY=4)
show_image_list([oc(image_filtered), oc(noisy_image)],
                ['Gaussian Blurred Image', 'Noisy Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_58_0.svg)
    


Sigma is like mean of the filter. Lets change it to see the effect of that


```python
image_filtered1 = cv2.GaussianBlur(noisy_image,
                                   ksize=(11, 11),
                                   sigmaX=10,
                                   sigmaY=10)
show_image_list([oc(noisy_image), oc(image_filtered), oc(image_filtered1)],
                ['Noisy Image', 'GaussianBlur \nsigma =4 ','GaussianBlur \n$sigma =10$' ])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_60_0.svg)
    


### Image Sharpening


```python
# Common filtering for image sharpening
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpen_cv = cv2. filter2D(lena_cv, -1, kernel)
show_image_list([oc(lena_cv), oc(sharpen_cv)],
                ['Original Image', 'Sharpen Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_62_0.svg)
    


## Edge Detection:PIL

* What is edge of an image ? Mathematically edges are where the pixel intensities change.
* We can approximate the gradient of grayscale image with convolution with convolution.


```python
barbara_pil = Image.open('barbara.png') 
# At firtst we enhance the edge so that 
# they can better picked up
barbara_edge_enhance = barbara_pil.filter(ImageFilter.EDGE_ENHANCE)
# Now finds the edges
barbara_edge = barbara_edge_enhance.filter(ImageFilter.FIND_EDGES)
show_image_list([barbara_pil,
                 barbara_edge_enhance,
                 barbara_edge],
                 ['Original Image',
                 'Enhanced Edge',
                 'Edged Image'],
                 ['gray','gray','gray','gray'])

```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_65_0.svg)
    


## Edge Detection:OpenCv


```python
barbara_cv = cv2.imread('barbara.png',
                        cv2.IMREAD_GRAYSCALE)
# We first smooth the image so that edge
# can be detected perfectly, otherwsie 
# there is chance that some noise will be there
barbara_smooth = cv2.GaussianBlur(barbara_cv, 
                                ksize=(3,3),
                                sigmaX=0.1,
                                sigmaY=0.1)
show_image_list([oc(barbara_cv),
                oc(barbara_smooth)],
                ['Orginal Image','Smoothed Image'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_67_0.svg)
    



* There are several methods for edge detection. We will use `Sobel edge detector`, which is actually combines several convolutions and finding the magnitude of the result

* The approximation of derviation can be done by using `Sobel` function. 
* Parameters
  * `src`: image
  * `ddepth`: output iamge depth, in case of 8 bit input image, it will  result truncated derivatives
  * `dx`: order of derivative x
  * `dy`: order of derivative y
  * `ksize`: extended sobel kernel: msut 1, 3, 5, or  7

dx = 1 represents the derivative in the x-direction.  The function approximates  the derivative by  convolving   the image with the following kernel


$\begin{bmatrix}
1 & 0 & -1 \\\\
2 & 0 & -2 \\\\
1 & 0 & -1
\end{bmatrix}$


```python
ddepth = cv2.CV_16S
# Apply this filter on the image in X direction
grad_x = cv2.Sobel(src=barbara_smooth,ddepth=ddepth,
                    dx=1,dy=0, ksize=3)
show_image_list([(barbara_smooth), (grad_x)],
                  ['Smoothed Image', ' Edge Approximation x direction'],
                  [ 'gray','gray' ])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_72_0.svg)
    


dy=1 represents the derivative in the y-direction.  The function approximates  the derivative by  convolving   the image with the following kernel
$\begin{bmatrix}
\ \ 1 & \ \ 2 & \ \ 1 \\\\
\ \ 0 & \ \ 0 & \ \ 0 \\\\
\ \ -1 & -2 & -1
\end{bmatrix}$


```python
grad_y = cv2.Sobel(src=barbara_smooth,ddepth=ddepth,
                    dx=0,dy=1, ksize=3)
show_image_list([(barbara_smooth), (grad_x), grad_y],
                  ['Smoothed Image', ' Edge Approximation x direction',' Edge Approximation y direction'],
                  [ 'gray','gray','gray' ])

```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_74_0.svg)
    



```python
# Converts the values back to a number  0 to 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Apply the function addWEighted to calcualte 
# the sum or two arrays 
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

```


```python

show_image_list([(barbara_smooth), (grad_x), grad_y, grad],
                  ['Smoothed Image', ' Edge Approximation x direction',' Edge Approximation y direction','Edge detection'],
                  [ 'gray','gray','gray','gray' ])

```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_76_0.svg)
    


## Median: PIL

Medial filters find the median of all pixels under the kernel area and the center is replaced with the median value. It will help in segmentation task, as it blurs the background




```python
cameraman = Image.open('cameraman.jpeg')
cameraman_median = cameraman.filter(ImageFilter.MedianFilter)
show_image_list([cameraman, cameraman_median],
                ['Oringinal','Medianfilter'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_79_0.svg)
    


## Median:Opencv


```python
cameraman_cv = cv2.imread('cameraman.jpeg', cv2.IMREAD_GRAYSCALE)
# blurring with kernal size= 5
camera_median = cv2.medianBlur(cameraman_cv, 5)
show_image_list([cameraman_cv, camera_median],
                ['Original','blurred'],
                ['gray','gray'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_81_0.svg)
    



```python
ret, outs = cv2.threshold(src=camera_median,
                          thresh=0,
                          maxval=255,
                          type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
ret, outs1 = cv2.threshold(src=cameraman_cv,
                          thresh=0,
                          maxval=255,
                          type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
show_image_list([cameraman_cv, outs, outs1],
                ['Origanl','Segmenation from blurred image','Segmenatation from actual image'],['gray','gray','gray'])
```


    
![svg](/images/Spatial_filtering_using_PIL_Opencv_files/Spatial_filtering_using_PIL_Opencv_82_0.svg)
    


* cv2.THRESH_OTSU is used, as we don't want to select manual threshold value
* cv2.THRESH_BINARY_INV is used to invert the binarasation, means (0 back o to 255 white)



References:
* This whole notebook is actually combination of two notebooks, which is my class note of coursera course. [Basic compuer vision ](https://www.coursera.org/learn/introduction-computer-vision-watson-opencv/home/welcome)
*   <a href='https://pillow.readthedocs.io/en/stable/index.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkCV0101ENCoursera25797139-2021-01-01'>Pillow Docs</a>

*  <a href='https://opencv.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkCV0101ENCoursera25797139-2021-01-01'>Open CV</a>
