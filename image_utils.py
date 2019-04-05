import cv2
import numpy as np
import matplotlib.pyplot as plt

# Using the code from the lesson we can define the following functions:

# Absolute thresholding based on the gradient
def abs_sobel_threshold(img, orient='x', thresh= (0, 255)):
    """
    Function that takes in an image and returns a binarized image based on hard
    thresholding of the result of the sobel operator
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Function that returns a binarized image based on gradient magnitude of the image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    # Return the binary image
    return binary_output

# Define a function that return a binarized output based on the gradient direction
# threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Function that takes in an image and thresholds and return a gradient directions
    for an image
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

def S_thresholding(img, thresh = (100,255)):
    """
    Function that takes in an image and return
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Get the S channel
    S = hls[:,:,2]
    # Binarize S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


# Create function to create a thresholded image using some of the function above
def generate_binarized(img, ksize = 3, absolute = True, abs_thresh = (150,255),
                                       magnitude = True, mag_thresh = (150,255),
                                       direction = True, dir_thresh = (0,np.pi/2),
                                       color_thresh = (100,255)):
    """
    Function that takes an image as input and return a binarized image
    using the following pipeline :
        - Apply a hard thresholding on the **Sobel operator**
        - Apply a gradient magnitude thresholding
        - Apply gradient direction thresholding
        - Transform image in the HLS color space and apply S channel thresholding
    """
    gradx = abs_sobel_threshold(img, orient='x', thresh=abs_thresh)
    grady = abs_sobel_threshold(img, orient='y', thresh=abs_thresh)
    mag_binary = mag_threshold(img, sobel_kernel=ksize, thresh= mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(gradx)

    # I chose to implement the if else regarding the choice of the thresholds after
    # computing all elements.
    # It is not optimized, but the fastest to implement. Will come back to it if needed.
    if absolute == True and magnitude == False and direction == False:
        combined[((gradx == 1) & (grady == 1))] = 1
    if absolute == False and magnitude == True and direction == False:
        combined[(mag_binary == 1)] = 1
    if absolute == False and magnitude == False and direction == True:
        combined[(dir_binary == 1)] = 1
    if absolute == True and magnitude == True and direction == False:
        combined[((gradx == 1) & (grady == 1)) | (mag_binary == 1)] = 1
    if absolute == True and magnitude == False and direction == True:
        combined[((gradx == 1) & (grady == 1)) | (dir_binary == 1)] = 1
    if absolute == False and magnitude == True and direction == True:
        combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    if absolute == True and magnitude == True and direction == True:
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Now, let us add some color thresholding using the HLS color space
    # Based on the lesson, as well as my impression I decided to keep using the
    # S channel
    s_contrib = S_thresholding(img, thresh = color_thresh)

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    contribs_binary = np.dstack(( np.zeros_like(s_contrib), s_contrib, combined)) * 255

    # Combine the two binary thresholds
    grad_color_binary = np.zeros_like(s_contrib)
    grad_color_binary[(s_contrib == 1) | (combined == 1)] = 1

    return contribs_binary, grad_color_binary


def warper(img, source_points, destination_points):
    """
    Function that returns a warped image based on the original image,
    the source points, and destination points.
    """
    M = cv2.getPerspectiveTransform(source_points, destination_points) # Compute the transformation matrix
    Minv = cv2.getPerspectiveTransform(destination_points, source_points) # Compute the transformation matrix

    warped_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])) # Image warping
    return warped_img, M , Minv


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
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
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

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

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

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 150

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
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
    _, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(binary_warped)


    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
#     window_img = np.zeros_like(out_img)
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
#                               ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
#                               ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#     # Draw the lane onto the warped blank image
# #    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#     ## End visualization steps ##

    return  left_fitx, left_fit, right_fitx, right_fit

def lane_back_to_image(image, warped_img, left_fitx, right_fitx, ploty, Minv, left_curvature, right_curvature, center):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
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
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)    
    cv2.putText(result, 'Radius of curvature = %d (m)'%((left_curvature+ right_curvature)/2), 
                (50,50),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),2, cv2.LINE_AA)
    if center > 0:
        direction = 'right'
    else:
        direction = 'left'

    cv2.putText(result, 'Vehicule is %.2f m %s of center'%(abs(center), direction), 
                (50,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),2,cv2.LINE_AA)
    return result
