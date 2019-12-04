#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'qt')

def camera_calibration():
	nx=9
	ny=6
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
	#print(objp)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	
	# Make a list of calibration images
	images = glob.glob('camera_cal/calibration*.jpg')
	
	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
		# Finding the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
		# adding object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	return ret,mtx,dist


# In[58]:


def thresh_binaryImage_old(image,ret,mtx,dist):
	dst = cv2.undistort(image, mtx, dist, None, mtx)
	# Convert to HLS color space and separate the S channel
	# Note: img is the undistorted image
	hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	# Grayscale image
	# NOTE: we already saw that standard grayscaling lost color information for the lane lines
	# Explore gradients in other colors spaces / color channels to see what might work better
	gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

	# Sobel x
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	thresh_min = 20
	thresh_max = 100
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Threshold color channel
	s_thresh_min = 170
	s_thresh_max = 255
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1  
	
	return combined_binary


# In[59]:


def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 15, thresh_min = 20, thresh_max = 120):
    # Grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply cv2.Sobel()
    sobel = 0
    gray = img

    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif (orient == 'y'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel)

    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Apply lower and upper thresholds
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Create binary_output
    return binary_output


def mag_thresh(img, sobel_kernel = 3, mag_thresh = (30, 100)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    gray = img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # 3) Calculate the magnitude 
    abs_sobelx  = np.sqrt(pow(sobelx, 2))
    abs_sobely  = np.sqrt(pow(sobely, 2))
    abs_sobelxy = np.sqrt(pow(sobelx, 2) + pow(sobely, 2))

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel = 15, thresh = (np.pi/4, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img

    # 2) Take the gradient in x and y separately
    #sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 15)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 15)


    # 3) Take the absolute value of the x and y gradients
    abs_sobelx  = np.sqrt(pow(sobelx, 2))
    abs_sobely  = np.sqrt(pow(sobely, 2))

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


# In[60]:


def combine_color_thresholds(img):
    #Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    #The provided image should be in RGB format
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Select saturation
    l_channel = hls[ : , : , 1]    
    # Threshold s chnnel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 110) & (l_channel <= 255)] = 1    
    # Select saturation
    s_channel = hls[ : , : , 2]    
    # Threshold s chnnel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 120) & (s_channel <= 255)] = 1    
    # Combine thresholds
    color_thresh = np.zeros_like(s_binary)    
    # Used the OR operation
    color_thresh[(s_binary == 1) & (l_binary == 1)] = 1
    return color_thresh  
    
def combine_thresholds(img, ksize = 7):
    # X, Y gradiets, magnitude of gradient and direction of gradient thresholds
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel = 15, thresh_min = 20, thresh_max = 120) # was 20-100
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel = 15, thresh_min = 20, thresh_max = 120)
    mag_binary = mag_thresh(img, sobel_kernel = 15, mag_thresh = (80, 200)) # was 30-100
    dir_binary = dir_threshold(img, sobel_kernel = ksize, thresh=(0, np.pi/2))

    # Combine thresholds
    #combined = np.zeros_like(gradx)
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) & ((grady == 1) | (mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1))] = 1
    
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    return combined

def thresh_binaryImage(image,ret,mtx,dist):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hls_thresh = combine_color_thresholds(img)
    undist_img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:,:,0]
    grad_thresh = combine_thresholds(undist_img_gray)
    # Combine thresholds
    combined = np.zeros_like(hls_thresh)
    
    # Combine with OR
    combined[(hls_thresh == 1) | (grad_thresh == 1)] = 1
    
    return combined  


# In[61]:


def Perspective_Transform(img_size,combined_binary):
	src = np.float32([[567,465], [745,465] , [1103,665] , [251,665]])  
	offset = 50
	dest = np.float32([[offset, offset], [img_size[0]-offset, offset], 
										[img_size[0]-offset, img_size[1]-offset], 
										[offset, img_size[1]-offset]]) 
	M = cv2.getPerspectiveTransform(src, dest)
	Minv=cv2.getPerspectiveTransform(dest, src)
	warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
	
	return M,Minv,warped


# In[62]:


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #print(nonzero)
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #print('leftL',left_lane_inds)
        #print('rightL',right_lane_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        #print('currentLX',leftx_current)
        #print('currentRX',rightx_current)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img


# In[63]:


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    #print(left_fit)
    right_fit = np.polyfit(righty, rightx, 2)
    #print(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    return left_fit,right_fit


# In[64]:


def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    #print(ploty)
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty


# In[65]:


def search_around_poly(binary_warped,left_fit,right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return left_fitx,right_fitx,ploty


# In[66]:


def generate_curve_eqtn(left_fitx,right_fitx,ym_per_pix, xm_per_pix):
    # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    return ploty, left_fit_cr, right_fit_cr


# In[67]:


def measure_curvature_real(left_fitx,right_fitx,img_size,left_fit,right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty, left_fit_cr, right_fit_cr = generate_curve_eqtn(left_fitx,right_fitx,ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = np.sqrt((1+(((2*left_fit_cr[0]*y_eval*ym_per_pix)+left_fit_cr[1])**2))**3)/np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad =np.sqrt((1+(((2*right_fit_cr[0]*y_eval*ym_per_pix)+right_fit_cr[1])**2))**3)/np.absolute(2*right_fit_cr[0])  ## Implement the calculation of the right line here
    
    yvalue = img_size[1]
    # Compute distance in meters of vehicle center from the line
    car_center = img_size[0]/2  # we assume the camera is centered in the car
    lane_center = ((left_fit[0]*yvalue**2 + left_fit[1]*yvalue + left_fit[2]) + (right_fit[0]*yvalue**2 + right_fit[1]*yvalue + right_fit[2])) / 2
    center_dist = (lane_center - car_center) * xm_per_pix
    return (left_curverad, right_curverad, center_dist)


# In[68]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


ret,mtx,dist=camera_calibration()
def process_image(image):
    undist = np.copy(image)
    img_size = (image.shape[1], image.shape[0])
    #combined_binary=thresh_binaryImage_old(image,ret,mtx,dist)
    combined_binary=thresh_binaryImage(image,ret,mtx,dist)
    M,Minv,warped=Perspective_Transform(img_size,combined_binary)
    left_fit,right_fit=fit_polynomial(warped)
    left_fitx,right_fitx,ploty=search_around_poly(warped,left_fit,right_fit)
    left_curverad, right_curverad,center_dist=measure_curvature_real(left_fitx,right_fitx,img_size,left_fit,right_fit)
    curvature = ((left_curverad+right_curverad)/2)
	# Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
	
	# Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    text='Curvature of lane = '+str(format(curvature, '.2f'))+'m'
    cv2.putText(result, text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
    if center_dist > 0:
        text = 'Vehicule position: '+str(format(center_dist, '.2f'))+ 'm left of center'
    else:
        text = 'Vehicule position: '+str(format(center_dist, '.2f'))+ 'm right of center'
    cv2.putText(result, text, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    return result


# In[69]:


output = 'output_videos/P2_Output.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(38,43)
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')
video_clip.reader.close()
video_clip.audio.reader.close_proc()


# In[ ]:


output = 'output_videos/P2_Output_challenge_video.mp4'
##clip1 = VideoFileClip("project_video.mp4").subclip(0,1)
clip2 = VideoFileClip("challenge_video.mp4")
video_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')
video_clip.reader.close()
video_clip.audio.reader.close_proc()


# In[ ]:


output = 'output_videos/P2_Output_harder_challenge_video.mp4'
##clip1 = VideoFileClip("project_video.mp4").subclip(0,1)
clip3 = VideoFileClip("harder_challenge_video.mp4")
video_clip = clip3.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')
video_clip.reader.close()
video_clip.audio.reader.close_proc()


# In[ ]:




