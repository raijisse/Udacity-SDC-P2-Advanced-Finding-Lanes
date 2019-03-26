import cv2
import numpy as np


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
