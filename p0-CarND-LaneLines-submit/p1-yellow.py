
# coding: utf-8

# In[2]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')
import math


# In[3]:


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count #?????
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[4]:


#Lane Extrapolation

def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
  left_lines, right_lines = [], []
  for line in lines:
    for x1, y1, x2, y2 in line:
      k = (y2 - y1) / (x2 - x1)
      if k < 0:
        left_lines.append(line)
      else:
        right_lines.append(line)
  
  if (len(left_lines) <= 0 or len(right_lines) <= 0): #if no lines, do nothing
    return img
  
  clean_lines(left_lines, 0.1) #slope diff threahold=0.1
  clean_lines(right_lines, 0.1) 
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line] #cranewhy
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
  
  left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
  right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
  
  cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
  cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
  
def clean_lines(lines, threshold):
  slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
  while len(lines) > 0:
    mean = np.mean(slope)
    diff = [abs(s - mean) for s in slope]
    idx = np.argmax(diff)
    if diff[idx] > threshold:
      slope.pop(idx)
      lines.pop(idx)
    else:
      break
  
  
def calc_lane_vertices(point_list, ymin, ymax):
  x = [p[0] for p in point_list]
  y = [p[1] for p in point_list]
  fit = np.polyfit(y, x, 1) #（fit x=[],y=[]）
  fit_fn = np.poly1d(fit)
  
  xmin = int(fit_fn(ymin))
  xmax = int(fit_fn(ymax))
  
  return [(xmin, ymin), (xmax, ymax)]


# In[5]:


#test on images

import os

path="test_images/"
files =os.listdir(path)
i=0
for file in files:
    i+=1
    image = mpimg.imread(path+"/"+file)#cranekey
    copy=np.copy(image)
    imshape = image.shape
    
    #
    gray_img=grayscale(copy)
    
    #gaussian parametres
    kernel_size=5
    gaussian_img=gaussian_blur(gray_img, kernel_size)
    # canny parametres
    low_threshold=50
    high_threshold=150
    canny_img=canny(gaussian_img, low_threshold, high_threshold)
    
    #region mask parametres
    vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    region_img=region_of_interest(canny_img, vertices)
    
    #Hough parametres
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    hough_lines_img=hough_lines(region_img, rho, theta, threshold, min_line_len, max_line_gap)
    
    
    #draw_lines(hough_lines_img, lines, minLineLength=min_line_len, maxLineGap=max_line_gap), color=[255, 0, 0], thickness=6)
    
    #add weighted
    final_weighted_img=weighted_img(hough_lines_img,image, α=0.8, β=1., λ=0.)
    
    
    plt.subplot(2,3,i)
    print(file)
    plt.imshow(final_weighted_img)


# In[6]:


#test on videos

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[14]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    #
    gray_img=grayscale(image)
    
    #
    kernel_size=5
    gaussian_img=gaussian_blur(gray_img, kernel_size)
    #
    low_threshold=50
    high_threshold=150
    canny_img=canny(gaussian_img, low_threshold, high_threshold)
    #
    #vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(80,imshape[0]),(450, 300), (480, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    region_img=region_of_interest(canny_img, vertices)
    
    #
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    hough_lines_img=hough_lines(region_img, rho, theta, threshold, min_line_len, max_line_gap)
    

    
    
    #draw_lines(hough_lines_img, lines, color=[255, 0, 0], thickness=6)
    
    
    final_weighted_img=weighted_img(hough_lines_img,image, α=0.8, β=1., λ=0.)
    
    
    return final_weighted_img


# In[17]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
yellow_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[18]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

