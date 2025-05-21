#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor # User confirmed this import works
from cv_bridge import CvBridge

bridge = CvBridge()
image = None
motor_pub = None
frame_count = 0
prev_angle = 0.0
white_lost_count = 0
prev_left_fit_coeffs = None
prev_right_fit_coeffs = None
YM_PER_PIX = 30.0 / 720.0 # Default, will be updated in start() based on warped_image_height
# XM_PER_PIX will be calculated dynamically. STANDARD_LANE_WIDTH_M / detected_lane_width_px
STANDARD_LANE_WIDTH_M = 3.7 # Standard lane width in meters for XM_PER_PIX calculation


# --------------------------------------------------------------------
# Image Preprocessing Helper Functions
# --------------------------------------------------------------------

# --- Visualization Function ---
def draw_lane_and_info(original_resized_bgr, binary_warped_img_shape, Minv_perspective, 
                       left_fitx, right_fitx, ploty, 
                       left_radius_m, right_radius_m, vehicle_offset_m):
    """
    Draws the detected lane area back onto the original image and displays curvature/offset.
    """
    final_image = original_resized_bgr.copy() # Start with the original image

    # Handle cases where lane lines might not be detected or fits are empty
    if left_fitx.size == 0 or right_fitx.size == 0 or ploty.size == 0:
        # Draw text info even if lane area cannot be drawn
        cv2.putText(final_image, f"L.Curv: {left_radius_m:.0f}m R.Curv: {right_radius_m:.0f}m", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA) # Black text
        cv2.putText(final_image, f"Offset: {vehicle_offset_m:.2f}m (No Lane Lines)", 
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        return final_image

    # Create an image to draw the lines on
    warp_zero = np.zeros(binary_warped_img_shape, dtype=np.uint8) # Use shape of binary_warped_img
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right_for_fill = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right_for_fill))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0)) # Green lane area

    # Optional: Draw lane lines themselves (thicker)
    cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255,0,0), thickness=10) # Blue for left
    pts_right_for_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))]) # No flip for polyline
    cv2.polylines(color_warp, np.int_([pts_right_for_line]), isClosed=False, color=(0,0,255), thickness=10) # Red for right


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    original_size = (original_resized_bgr.shape[1], original_resized_bgr.shape[0])
    newwarp = cv2.warpPerspective(color_warp, Minv_perspective, original_size) 
    
    # Combine the result with the original image
    final_image = cv2.addWeighted(original_resized_bgr, 1, newwarp, 0.3, 0)

    # Add text overlays for curvature and offset
    curvature_text = f"L.Curv: {left_radius_m:.0f}m R.Curv: {right_radius_m:.0f}m"
    if left_radius_m == float('inf') and right_radius_m == float('inf'):
        curvature_text = "Curvature: Straight"
    elif left_radius_m == float('inf'):
        curvature_text = f"R.Curv: {right_radius_m:.0f}m"
    elif right_radius_m == float('inf'):
        curvature_text = f"L.Curv: {left_radius_m:.0f}m"
        
    cv2.putText(final_image, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA) # White text
    cv2.putText(final_image, f"Offset: {vehicle_offset_m:.2f}m", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return final_image

# --- Curvature and Offset Calculation Functions ---
def measure_curvature_real(ploty_px, left_fit_coeffs_px, right_fit_coeffs_px, H_warped_px, current_xm_per_pix):
    """
    Calculates the curvature of polynomial functions in meters.
    """
    global YM_PER_PIX # Use the global YM_PER_PIX updated in start()
    left_curverad_m = float('inf')
    right_curverad_m = float('inf')

    if ploty_px is None or ploty_px.size == 0:
        return left_curverad_m, right_curverad_m

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_px = np.max(ploty_px)
    y_eval_m = y_eval_px * YM_PER_PIX

    if left_fit_coeffs_px is not None:
        # Recalculate fit in terms of meters
        leftx_px = left_fit_coeffs_px[0]*ploty_px**2 + left_fit_coeffs_px[1]*ploty_px + left_fit_coeffs_px[2]
        left_fit_cr_m = np.polyfit(ploty_px * YM_PER_PIX, leftx_px * current_xm_per_pix, 2)
        A_m_left = left_fit_cr_m[0]
        B_m_left = left_fit_cr_m[1]
        if A_m_left != 0: # Avoid division by zero if line is straight
            left_curverad_m = ((1 + (2*A_m_left*y_eval_m + B_m_left)**2)**1.5) / np.absolute(2*A_m_left)
        # Else, curvature is infinite (straight line)

    if right_fit_coeffs_px is not None:
        rightx_px = right_fit_coeffs_px[0]*ploty_px**2 + right_fit_coeffs_px[1]*ploty_px + right_fit_coeffs_px[2]
        right_fit_cr_m = np.polyfit(ploty_px * YM_PER_PIX, rightx_px * current_xm_per_pix, 2)
        A_m_right = right_fit_cr_m[0]
        B_m_right = right_fit_cr_m[1]
        if A_m_right != 0: # Avoid division by zero
            right_curverad_m = ((1 + (2*A_m_right*y_eval_m + B_m_right)**2)**1.5) / np.absolute(2*A_m_right)
        # Else, curvature is infinite (straight line)
        
    return left_curverad_m, right_curverad_m

def calculate_vehicle_offset(W_warped_px, left_fitx_bottom_px, right_fitx_bottom_px, current_xm_per_pix):
    """
    Calculates the vehicle offset from the center of the lane.
    Offset is positive if vehicle is to the right of center, negative if to the left.
    """
    if left_fitx_bottom_px is None or right_fitx_bottom_px is None:
        return 0.0 # Cannot determine offset if one lane is missing at the bottom

    vehicle_center_px = W_warped_px / 2.0
    lane_center_px = (left_fitx_bottom_px + right_fitx_bottom_px) / 2.0
    offset_px = vehicle_center_px - lane_center_px # Positive if vehicle_center > lane_center (vehicle to the right of lane center)
    offset_m = offset_px * current_xm_per_pix
    return offset_m

# --- Search Around Polynomial Function ---
def search_around_poly(binary_warped, current_left_fit, current_right_fit, margin=80):
    """
    Finds lane pixels by searching around previously fitted polynomial lines.
    Args:
        binary_warped: Bird's-eye view binary image.
        current_left_fit: Polynomial coefficients for the left lane from a previous fit.
        current_right_fit: Polynomial coefficients for the right lane from a previous fit.
        margin: Half-width of the search area around the polynomial lines.
    Returns:
        leftx, lefty, rightx, righty: Pixel coordinates for left and right lane lines.
        out_img: Image with detected lane pixels drawn.
    """
    if current_left_fit is None and current_right_fit is None:
        # Cannot search if no prior fits are available
        out_img_blank = np.dstack((binary_warped, binary_warped, binary_warped)) if binary_warped.ndim == 2 else binary_warped.copy()
        if np.max(binary_warped) == 1 and binary_warped.ndim == 2 : # If input is 0-1 binary
             out_img_blank = out_img_blank * 255
        return np.array([]), np.array([]), np.array([]), np.array([]), out_img_blank.astype(np.uint8)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) if binary_warped.ndim == 2 else binary_warped.copy()
    if np.max(binary_warped) == 1 and binary_warped.ndim == 2 : # If input is 0-1 binary
        out_img = out_img * 255
    out_img = out_img.astype(np.uint8)


    left_lane_inds = np.array([], dtype=np.int32)
    right_lane_inds = np.array([], dtype=np.int32)

    if current_left_fit is not None:
        left_fit_line_x = current_left_fit[0]*(nonzeroy**2) + current_left_fit[1]*nonzeroy + current_left_fit[2]
        left_lane_inds_bool = (nonzerox >= left_fit_line_x - margin) & (nonzerox < left_fit_line_x + margin)
        left_lane_inds = left_lane_inds_bool.nonzero()[0]


    if current_right_fit is not None:
        right_fit_line_x = current_right_fit[0]*(nonzeroy**2) + current_right_fit[1]*nonzeroy + current_right_fit[2]
        right_lane_inds_bool = (nonzerox >= right_fit_line_x - margin) & (nonzerox < right_fit_line_x + margin)
        right_lane_inds = right_lane_inds_bool.nonzero()[0]

    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([]) 
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    if len(leftx) > 0:
        out_img[lefty, leftx] = [255, 0, 0] # Red
    if len(rightx) > 0:
        out_img[righty, rightx] = [0, 0, 255] # Blue
        
    return leftx, lefty, rightx, righty, out_img

# --- Polynomial Fitting Function ---
def fit_polynomial(leftx, lefty, rightx, righty, warped_height):
    """
    Fits a 2nd order polynomial to left and right lane pixel coordinates.
    Args:
        leftx, lefty: X and Y coordinates of left lane pixels.
        rightx, righty: X and Y coordinates of right lane pixels.
        warped_height: Height of the warped image (for generating ploty).
    Returns:
        left_fit: Coefficients of the polynomial for the left lane.
        right_fit: Coefficients of the polynomial for the right lane.
        left_fitx: X values of the fitted polynomial for the left lane.
        right_fitx: X values of the fitted polynomial for the right lane.
        ploty: Y values corresponding to left_fitx and right_fitx.
    """
    left_fit = None
    right_fit = None
    left_fitx = np.array([])
    right_fitx = np.array([])

    ploty = np.linspace(0, warped_height - 1, warped_height)

    # Fit a second order polynomial to pixel positions in each fake lane line
    if len(lefty) > 2 and len(leftx) > 2 : # Need at least 3 points to fit a 2nd order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    if len(righty) > 2 and len(rightx) > 2: # Need at least 3 points
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    return left_fit, right_fit, left_fitx, right_fitx, ploty

# --- Sliding Window Lane Finding Function ---
def find_lane_pixels_sliding_window(binary_warped, nwindows=10, margin=100, minpix=50):
    """
    Finds lane pixels using the sliding window method.
    Args:
        binary_warped: Bird's-eye view binary image.
        nwindows: Number of sliding windows.
        margin: Half-width of the windows.
        minpix: Minimum number of pixels found to recenter window.
    Returns:
        leftx, lefty, rightx, righty: Pixel coordinates for left and right lane lines.
        out_img: Image with sliding windows and detected lane pixels drawn.
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    if binary_warped.ndim == 2: # Ensure 3 channels for color drawing
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    else:
        out_img = binary_warped.copy() # Assuming it's already 3-channel if not 2D

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Handle cases where no pixels were found in any window
        pass # Keep them as empty lists or empty np.arrays

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([]) 
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    # Color lane pixels
    if len(left_lane_inds) > 0:
        out_img[lefty, leftx] = [255, 0, 0] # Red
    if len(right_lane_inds) > 0:
        out_img[righty, rightx] = [0, 0, 255] # Blue

    return leftx, lefty, rightx, righty, out_img

# --- Perspective Transform Function ---
def warp_image(img_roi, src_points_roi, dst_points_transform, dsize_warp):
    """
    Applies a perspective transform to warp the image.
    Args:
        img_roi: Input image (Region of Interest).
        src_points_roi: np.float32 array of 4 source points in the ROI.
        dst_points_transform: np.float32 array of 4 destination points.
        dsize_warp: Tuple (width, height) for the output warped image size.
    Returns:
        warped_img: The perspective-warped image.
        M: The perspective transform matrix.
        Minv: The inverse perspective transform matrix.
    """
    M = cv2.getPerspectiveTransform(src_points_roi, dst_points_transform)
    Minv = cv2.getPerspectiveTransform(dst_points_transform, src_points_roi)
    warped_img = cv2.warpPerspective(img_roi, M, dsize_warp, flags=cv2.INTER_LINEAR)
    return warped_img, M, Minv

def apply_hls_thresholds(hls_image, thresh_h, thresh_l, thresh_s):
    """
    Applies H, L, S thresholds to an HLS image and combines them.
    Args:
        hls_image: Input image in HLS color space.
        thresh_h: Tuple (low, high) for H channel threshold.
        thresh_l: Tuple (low, high) for L channel threshold.
        thresh_s: Tuple (low, high) for S channel threshold.
    Returns:
        A binary image resulting from the combined thresholds.
    """
    h_channel = hls_image[:,:,0]
    l_channel = hls_image[:,:,1]
    s_channel = hls_image[:,:,2]

    # Apply thresholds
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= thresh_h[0]) & (h_channel <= thresh_h[1])] = 255

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1])] = 255
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 255

    # Combination logic from the blog post
    # combined_binary[((s_binary == 255) & (l_binary == 0)) | ((s_binary == 0) & (h_binary == 255) & (l_binary == 255))] = 255
    # Corrected logic based on common practice (usually ORing or ANDing specific conditions)
    # The blog's specific logic:
    # "흰색 영역: S 값이 높고(255) L 값은 낮은(0) 픽셀" -> s_binary AND (NOT l_binary)
    # "노란색 영역: S 값이 낮고(0) H 값과 L 값이 모두 높은(255) 픽셀" -> (NOT s_binary) AND h_binary AND l_binary
    # This seems to target very specific shades. Let's use the direct logic given.
    combined_binary = np.zeros_like(s_channel, dtype=np.uint8)
    
    # Condition from blog: ((s_binary == 255) & (l_binary == 0)) | ((s_binary == 0) & (h_binary == 255) & (l_binary == 255))
    # To apply this, we need to consider that l_binary is 255 where it meets its threshold, 0 otherwise.
    # So, "l_binary == 0" means l_channel is *outside* its threshold range.
    # For "l_binary == 255", it means l_channel is *inside* its threshold range.

    # Let's re-evaluate the blog's logic carefully:
    # 흰색 영역: S 값이 높고(thresh_s) L 값은 낮은(not thresh_l) 픽셀
    # 노란색 영역: S 값이 낮고(not thresh_s) H 값과 L 값이 모두 높은(thresh_h and thresh_l) 픽셀
    # The blog's combined_binary line implies specific values (0 or 255) rather than ranges.
    # For this implementation, we will use the provided complex condition directly on the binary masks.
    
    # Condition 1: (s_binary == 255) AND (l_binary == 0)
    # This means S is in its range, L is NOT in its range (e.g., L is low if thresh_l is for high L values)
    # If thresh_l = (50, 160), then l_binary==0 means l_channel < 50 or l_channel > 160.
    # The blog seems to interpret l_binary == 0 as "L is low".
    # The blog states: "L 값은 낮은(0) 픽셀" which is ambiguous.
    # Let's assume the blog means S is high AND L is low (e.g. L < some_low_fixed_value like 50)
    # OR S is low AND H is high AND L is high.
    # Given the direct formula: combined_binary[((s_binary == 255) & (l_binary == 0)) | ((s_binary == 0) & (h_binary == 255) & (l_binary == 255))] = 255
    # This means:
    # 1. s_channel is within its threshold AND l_channel is NOT within its threshold.
    # OR
    # 2. s_channel is NOT within its threshold AND h_channel IS within its threshold AND l_channel IS within its threshold.
    
    cond1 = (s_binary == 255) & (l_binary == 0) # S is high, L is out of its main range (potentially low or very high)
    cond2 = (s_binary == 0) & (h_binary == 255) & (l_binary == 255) # S is out of range, H is in range, L is in range
    
    combined_binary[cond1 | cond2] = 255
    
    return combined_binary

def apply_sobel_threshold(image_channel, orient='x', ksize=3, thresh=(20, 100)):
    """
    Applies Sobel derivative thresholding to a single image channel.
    Args:
        image_channel: Single channel grayscale image.
        orient: Orientation of the Sobel derivative ('x' or 'y').
        ksize: Kernel size for Sobel operator.
        thresh: Tuple (low, high) for thresholding the scaled Sobel derivative.
    Returns:
        A binary image.
    """
    if orient == 'x':
        sobel = cv2.Sobel(image_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv2.Sobel(image_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        raise ValueError("orient must be 'x' or 'y'")

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

def preprocess_pipeline(bgr_image, roi_coords):
    """
    Applies ROI slicing and HLS thresholding.
    Args:
        bgr_image: Input BGR image.
        roi_coords: Tuple (y_start, y_end, x_start, x_end) for ROI.
    Returns:
        A binary image processed for lane detection within the ROI.
    """
    y_start, y_end, x_start, x_end = roi_coords
    roi_image = bgr_image[y_start:y_end, x_start:x_end]

    hls_roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HLS)

    # Thresholds from the blog post (may need tuning)
    # H for yellow, L for general brightness, S for saturation (helps with white)
    th_h = (15, 35)    # Hue range for yellow (typical for yellow)
    th_l = (30, 200)   # Lightness range (broad to capture lines in shadows/highlights)
    th_s = (100, 255)  # Saturation range (high saturation for strong colors like yellow, also helps white)
    
    # The blog's HLS combination logic is specific:
    # combined_binary[((s_binary == 255) & (l_binary == 0)) | ((s_binary == 0) & (h_binary == 255) & (l_binary == 255))] = 255
    # For this to work as intended by the blog, the binary images need to be generated with ranges that make sense for the conditions.
    # Let's adjust the HLS thresholds to typical values for robust white/yellow detection first,
    # then attempt the blog's combination logic if direct combination is not good enough.
    # For now, using the blog's direct complex combination logic with these more standard ranges for individual channels.
    # The blog's example thresholds were: th_h, th_l, th_s = (160, 255), (50, 160), (0, 255)
    # These seem unusual (H=160-255 is more like magenta/red). Let's use the ones from the prompt for now.
    # Prompt's proposed thresholds: th_h, th_l, th_s = (160, 255), (50, 160), (0, 255)
    # Let's use these for apply_hls_thresholds as it expects it.
    # The blog describes: H(노란색), L(조명), S(흰색) -> H for yellow, L for illumination, S for white.
    # H (노란색): 15 ~ 35 (yellow range)
    # L (조명): 30 ~ 200 (general brightness)
    # S (흰색): 100 ~ 255 (high saturation for white under various light)
    # The formula given in the prompt for apply_hls_thresholds is:
    # combined_binary[((s_binary == 255) & (l_binary == 0)) | ((s_binary == 0) & (h_binary == 255) & (l_binary == 255))] = 255
    # This specific combination logic from the blog is what `apply_hls_thresholds` implements.
    # So, the thresholds passed to it should be the ones the blog intended for that formula.
    # Blog's values: H(160,255), L(50,160), S(0,255) -> These are passed.
    blog_th_h, blog_th_l, blog_th_s = (160, 255), (50, 160), (0, 255)
    hls_binary = apply_hls_thresholds(hls_roi_image, blog_th_h, blog_th_l, blog_th_s)

    # (Future/Alternative - Sobel)
    # gray_roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # l_channel_roi = hls_roi_image[:,:,1]
    # sobel_x_binary = apply_sobel_threshold(l_channel_roi, orient='x', ksize=3, thresh=(25, 100))
    # combined_output = cv2.bitwise_or(hls_binary, sobel_x_binary)
    # return combined_output
    
    return hls_binary


def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def drive(angle, speed):
    msg = XycarMotor()
    msg.angle = float(angle)
    msg.speed = float(speed)
    motor_pub.publish(msg)

def start():
    global motor_pub, frame_count, prev_angle, white_lost_count, prev_left_fit_coeffs, prev_right_fit_coeffs, YM_PER_PIX # image is already global

    rospy.init_node('xycar_lane_keeper') # Or a new name like 'xycar_lane_keeper'
    motor_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    
    rospy.loginfo("Waiting for image topics...")
    rospy.wait_for_message("/usb_cam/image_raw", Image) # Wait until the first image is received
    rospy.loginfo("Image received. Starting advanced lane keeping node.")
    
    # Original code had rospy.sleep(1.0) here, wait_for_message is more robust

    print("▶▶▶ Advanced Lane Detection Mode Start") # Updated print message

    while not rospy.is_shutdown():
        if image is None:
            rospy.logwarn_throttle(1.0, "Image is None, skipping frame.")
            time.sleep(0.1) # Prevent busy-looping if image is consistently None
            continue

        # Resize image (as per blog example)
        original_height, original_width = image.shape[:2]
        proc_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        current_height, current_width = proc_image.shape[:2] # New height and width after resize
        
        display_image = proc_image.copy() # Image for drawing visualizations on

        # Conceptual ROI params for preprocessing functions (based on resized image dimensions)
        # These will be passed to and used by the actual preprocessing functions later.
        # Example: roi_y_start = int(current_height * 0.6)
        # Example: roi_y_end = int(current_height * 0.95)

        # Define ROI for preprocessing (using resized image dimensions)
        # These numbers are from the blog, relative to the *resized* image (e.g., 360p).
        roi_y_start = 220 
        roi_y_start = min(roi_y_start, current_height - 20) # Safeguard: ensure at least 20px height for ROI
        roi_y_end = current_height - 12
        
        if roi_y_end <= roi_y_start: # Ensure y_end > y_start
            roi_y_start = int(current_height * 0.5) # Fallback: upper half
            roi_y_end = current_height -1 # Fallback: near bottom
            if roi_y_end <= roi_y_start: # If still problematic (very small image)
                 roi_y_start = 0
                 roi_y_end = current_height


        roi_x_start = 0
        roi_x_end = current_width
        roi_coords_for_preprocess = (roi_y_start, roi_y_end, roi_x_start, roi_x_end)

        # --- STAGE 1: Image Preprocessing ---
        combined_binary_image = preprocess_pipeline(proc_image, roi_coords_for_preprocess)


        # --- STAGE 2: Perspective Transform ---
        M = None # Initialize to None
        Minv = None # Initialize to None

        if combined_binary_image is None or combined_binary_image.size == 0:
            rospy.logwarn_throttle(1.0, "Combined binary image is empty, skipping warp.")
            # Create a dummy black warped_image if preprocessing failed
            # Use last known good dimensions or a default if not available
            # For now, using a fixed default size for dummy image
            warped_image_width_default = 320
            warped_image_height_default = 400
            warped_image = np.zeros((warped_image_height_default, warped_image_width_default), dtype=np.uint8)
        else:
            roi_h, roi_w = combined_binary_image.shape[:2]

            # Source points for perspective transform (relative to combined_binary_image)
            # Using blog's second set, scaled by roi_w relative to a 640px reference.
            # Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left
            ref_width_for_blog_pts = 640.0 
            scale_x = roi_w / ref_width_for_blog_pts
            
            src_pts = np.float32([
                [100 * scale_x, roi_h],      # Bottom-left
                [450 * scale_x, roi_h],      # Bottom-right
                [310 * scale_x, 40],         # Top-right (Y=40 from top of ROI)
                [270 * scale_x, 40]          # Top-left (Y=40 from top of ROI)
            ])

            # Destination points and size for the warped image
            warped_image_width = 320
            warped_image_height = 400 # This is H_warped_px for YM_PER_PIX calculation
            dsize_warp = (warped_image_width, warped_image_height)
            
            # Update global YM_PER_PIX based on the actual warped image height used
            # Assuming the vertical span of the bird's-eye view covers approx 30 meters.
            YM_PER_PIX = 30.0 / warped_image_height


            dst_pts = np.float32([
                [0, warped_image_height],                  # Bottom-left
                [warped_image_width, warped_image_height], # Bottom-right
                [warped_image_width, 0],                   # Top-right
                [0, 0]                                     # Top-left
            ])

            warped_image, M, Minv = warp_image(combined_binary_image, src_pts, dst_pts, dsize_warp)

        # --- STAGE 3: Lane Pixel Identification ---
        nwindows = 10 
        margin = 80   
        minpix = 40   
        search_margin_around_poly = 80
        current_lane_finding_viz_img = None # To store the visualization image from the chosen method

        if prev_left_fit_coeffs is not None or prev_right_fit_coeffs is not None:
            # rospy.loginfo("Attempting search around polynomial.")
            left_x, left_y, right_x, right_y, poly_search_viz_img = \
                search_around_poly(warped_image, prev_left_fit_coeffs, prev_right_fit_coeffs, margin=search_margin_around_poly)
            
            if (len(left_x) < minpix or len(right_x) < minpix):
                # rospy.loginfo("Search around poly failed or found too few points, falling back to sliding window.")
                if np.any(warped_image):
                    left_x, left_y, right_x, right_y, sliding_window_viz_img = \
                        find_lane_pixels_sliding_window(warped_image, nwindows=nwindows, margin=margin, minpix=minpix)
                    current_lane_finding_viz_img = sliding_window_viz_img 
                else:
                    left_x, left_y, right_x, right_y = np.array([]), np.array([]), np.array([]), np.array([])
                    current_lane_finding_viz_img = np.dstack((warped_image, warped_image, warped_image)) if warped_image.ndim == 2 else warped_image.copy()
                    if np.max(warped_image) == 1 and warped_image.ndim == 2 : current_lane_finding_viz_img = current_lane_finding_viz_img * 255
                    current_lane_finding_viz_img = current_lane_finding_viz_img.astype(np.uint8)
            else:
                # rospy.loginfo("Search around poly successful.")
                current_lane_finding_viz_img = poly_search_viz_img
        else: 
            # rospy.loginfo("No previous fit, using sliding window.")
            if np.any(warped_image):
                left_x, left_y, right_x, right_y, sliding_window_viz_img = \
                    find_lane_pixels_sliding_window(warped_image, nwindows=nwindows, margin=margin, minpix=minpix)
                current_lane_finding_viz_img = sliding_window_viz_img
            else:
                left_x, left_y, right_x, right_y = np.array([]), np.array([]), np.array([]), np.array([])
                current_lane_finding_viz_img = np.dstack((warped_image, warped_image, warped_image)) if warped_image.ndim == 2 else warped_image.copy()
                if np.max(warped_image) == 1 and warped_image.ndim == 2 : current_lane_finding_viz_img = current_lane_finding_viz_img * 255
                current_lane_finding_viz_img = current_lane_finding_viz_img.astype(np.uint8)

        
        # --- STAGE 4: Polynomial Fitting ---
        current_warped_height = warped_image.shape[0] if warped_image is not None and hasattr(warped_image, 'shape') and warped_image.ndim == 2 else 0
        
        left_fit_coeffs, right_fit_coeffs = None, None
        left_fitx_plot, right_fitx_plot, ploty_generated = np.array([]), np.array([]), np.array([])

        if current_warped_height > 0 :
            left_fit_coeffs, right_fit_coeffs, left_fitx_plot, right_fitx_plot, ploty_generated = \
                fit_polynomial(left_x, left_y, right_x, right_y, current_warped_height)

        # Update previous fits if current fitting is successful
        if left_fit_coeffs is not None:
            prev_left_fit_coeffs = left_fit_coeffs
        # else: prev_left_fit_coeffs = None # Optional: reset if current fit fails

        if right_fit_coeffs is not None:
            prev_right_fit_coeffs = right_fit_coeffs
        # else: prev_right_fit_coeffs = None # Optional: reset if current fit fails


        # Temporary visualization of fitted polynomials on current_lane_finding_viz_img
        if current_lane_finding_viz_img is not None and current_lane_finding_viz_img.size > 0:
            if left_fit_coeffs is not None and left_fitx_plot.size > 0 and ploty_generated.size > 0:
                pts_left = np.array([np.transpose(np.vstack([left_fitx_plot, ploty_generated]))])
                cv2.polylines(current_lane_finding_viz_img, np.int_([pts_left]), isClosed=False, color=(255,255,0), thickness=2) # Cyan
            
            if right_fit_coeffs is not None and right_fitx_plot.size > 0 and ploty_generated.size > 0:
                pts_right = np.array([np.transpose(np.vstack([right_fitx_plot, ploty_generated]))])
                cv2.polylines(current_lane_finding_viz_img, np.int_([pts_right]), isClosed=False, color=(0,255,255), thickness=2) # Yellow
        
        
        # --- STAGE 5: Curvature & Offset Calculation ---
        left_curvature_m, right_curvature_m = float('inf'), float('inf')
        vehicle_offset_m = 0.0
        # Default xm_per_pix, assuming warped_image_width (e.g. 320px) corresponds to standard lane width
        # This will be dynamically updated if possible
        XM_PER_PIX_DEFAULT = STANDARD_LANE_WIDTH_M / warped_image_width 
        calculated_xm_per_pix = XM_PER_PIX_DEFAULT

        if ploty_generated.size > 0 and left_fitx_plot.size > 0 and right_fitx_plot.size > 0 and \
           left_fit_coeffs is not None and right_fit_coeffs is not None and \
           warped_image is not None and warped_image.shape[0] > 0 and warped_image.shape[1] > 0:
            
            bottom_left_x_px = left_fitx_plot[-1]
            bottom_right_x_px = right_fitx_plot[-1]
            
            lane_width_at_bottom_px = bottom_right_x_px - bottom_left_x_px
            if lane_width_at_bottom_px > 10: # Avoid division by zero or too small width
                calculated_xm_per_pix = STANDARD_LANE_WIDTH_M / lane_width_at_bottom_px
            # else: use XM_PER_PIX_DEFAULT already set

            left_curvature_m, right_curvature_m = measure_curvature_real(
                ploty_generated, left_fit_coeffs, right_fit_coeffs, 
                current_warped_height, calculated_xm_per_pix
            )

            vehicle_offset_m = calculate_vehicle_offset(
                warped_image.shape[1], bottom_left_x_px, bottom_right_x_px, 
                calculated_xm_per_pix
            )
            
            rospy.loginfo(f"L.Curv: {left_curvature_m:.0f}m, R.Curv: {right_curvature_m:.0f}m, Offset: {vehicle_offset_m:.2f}m, XMPX: {calculated_xm_per_pix:.4f}")
        else:
            rospy.logwarn("Skipping curvature/offset calculation due to missing fits or invalid data.")

        
        # --- STAGE 6: Visualization ---
        final_display_image = None # Initialize
        if 'Minv' in locals() and Minv is not None: # Minv is from Stage 2
            final_display_image = draw_lane_and_info(
                display_image,       # Resized original BGR image
                warped_image.shape,  # Pass shape of binary_warped_img
                Minv,
                left_fitx_plot,      # From Stage 4
                right_fitx_plot,     # From Stage 4
                ploty_generated,     # From Stage 4
                left_curvature_m,    # From Stage 5
                right_curvature_m,   # From Stage 5
                vehicle_offset_m     # From Stage 5
            )
        else: # Fallback if Minv is not available
            final_display_image = display_image.copy()
            # Draw default/error text if Minv was not available
            cv2.putText(final_display_image, f"L.Curv: {left_curvature_m:.0f}m R.Curv: {right_curvature_m:.0f}m", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA) # Red text
            cv2.putText(final_display_image, f"Offset: {vehicle_offset_m:.2f}m (Warp Failed)", 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)


        # --- STAGE 7: Steering and Speed Calculation (To be implemented) ---
        current_angle = 0.0  # Placeholder
        current_speed = 30.0 # Placeholder speed

        # Apply steering smoothing (kept from previous version)
        max_delta = 3.0
        if abs(current_angle - prev_angle) > max_delta:
            current_angle = prev_angle + np.sign(current_angle - prev_angle) * max_delta
        prev_angle = current_angle

        # Update the main log info, potentially including curvature/offset later
        rospy.loginfo(f"[CONTROL] Angle: {current_angle:.2f}, Speed: {current_speed:.2f}")


        cv2.imshow("Original Resized", display_image) # Show the resized image
        cv2.imshow("Processed Binary", combined_binary_image) 
        cv2.imshow("Warped Image", warped_image) 
        cv2.imshow("Lane Search Visualization", current_lane_finding_viz_img) # Updated window name
        cv2.imshow("Final Lane Detection", final_display_image) 
        cv2.waitKey(1)

        drive(current_angle, current_speed)
        time.sleep(0.02)

if __name__ == '__main__':
    try:
        start()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Cleanup OpenCV windows on exit
        cv2.destroyAllWindows()
