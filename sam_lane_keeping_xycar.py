#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor  # User-confirmed import
from cv_bridge import CvBridge

# --- Global Variables for ROS ---
bridge = CvBridge()
current_image_cv = None  # Stores the latest image from the camera in OpenCV format
motor_control_publisher = None # Publishes motor commands

# --- Global State Variables for Lane Detection ---
previous_left_polynomial_fit = None  # Stores coefficients of the left lane polynomial from the previous frame
previous_right_polynomial_fit = None # Stores coefficients of the right lane polynomial from the previous frame
previous_vehicle_offset_meters = 0.0 # Stores the vehicle's offset from the previous frame, for D-term calculation
last_time_for_derivative_calc = 0.0  # Stores the timestamp of the last D-term calculation
accumulated_integral_offset_meters = 0.0 # Stores the accumulated offset for the I-term

# --- Configuration Constants ---

# Image Processing
ROI_Y_START_PERCENTAGE = 0.55  # Pixels from top of RESIZED image to start ROI. Decrease to see further ahead in ROI.
ROI_Y_END_PERCENTAGE = 0.95    # End ROI at 95% of image height from the top
IMAGE_RESIZED_WIDTH = 320      # Width for image resizing
IMAGE_RESIZED_HEIGHT = 240     # Height for image resizing

# HLS Color Thresholds (example values, may need tuning)
H_THRESHOLD_YELLOW_LANE = (15, 35)    # Hue range for detecting yellow lanes
L_THRESHOLD_UNIVERSAL_LANE = (30, 200) # Lightness range for general lane visibility
S_THRESHOLD_WHITE_YELLOW_LANE = (100, 255) # Saturation range for white/yellow lanes

# Perspective Transform: Source points (relative to ROI of resized image)
# These define a trapezoid. Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
# MUST BE TUNED for the specific camera setup.
SRC_POINTS_ROI_RATIOS = np.float32([
    [0.40, 0.05],  # Top-left. Y-coord (0.05) is % of ROI height from ROI top. Smaller Y = further view.
    [0.60, 0.05],  # Top-right. Y-coord (0.05) is % of ROI height from ROI top. Smaller Y = further view.
    [0.95, 0.90],  # Bottom-right
    [0.05, 0.90]   # Bottom-left
])

# Perspective Transform: Destination Parameters
WARPED_IMAGE_WIDTH_AS_ROI_RATIO = 1.0  # Warped image width as a ratio of the ROI width
WARPED_IMAGE_HEIGHT_AS_ROI_RATIO = 2.5 # Ratio of warped image height to ROI height. Larger stretches view further.

# Sliding Window Algorithm Parameters
SLIDING_WINDOW_COUNT = 10
SLIDING_WINDOW_MARGIN_PERCENTAGE = 0.15 # Window margin as percentage of the warped image width
MIN_PIXELS_FOR_RECENTERING = 50

# Polynomial Search Margin (when using a previous fit)
POLYNOMIAL_SEARCH_MARGIN_PERCENTAGE = 0.10 # Margin as percentage of the warped image width

# Pixel-to-Meter Conversion Factors (Example values, MUST BE TUNED based on camera calibration and warped image dimensions)
# These are critical for converting pixel measurements to real-world distances for curvature and offset.
YM_PER_PIX_WARPED = 30.0 / 720.0  # Example: 30 meters visible vertically maps to 720 pixels in warped image height
XM_PER_PIX_WARPED = 3.7 / 700.0   # Example: Standard 3.7m lane width maps to 700 pixels in warped image width

# PID and Feed-Forward Controller Gains (CRITICAL: NEEDS EXTENSIVE TUNING)
# =====================================================================================
# These are initial placeholder values and WILL require significant tuning 
# based on the specific vehicle dynamics (e.g., Xycar in Unity simulator or real hardware) 
# and the characteristics of the track / environment.
# =====================================================================================
KP_LATERAL_OFFSET = 30.0      # Proportional gain for vehicle's lateral offset from lane center
KI_LATERAL_OFFSET = 2.0       # Integral gain for lateral offset (helps eliminate steady-state error)
KD_LATERAL_OFFSET = 10.0      # Derivative gain for lateral offset (helps dampen oscillations)
KP_HEADING_ERROR = 25.0       # Proportional gain for heading error (angle between car and lane)
K_FF_LANE_CURVATURE = 0.8     # Feed-forward gain for lane curvature (uses Am coefficient of centerline polynomial)
MAX_INTEGRAL_TERM_VALUE = 0.5 # Anti-windup limit for the integral term of offset

# Steering and Speed Limits
MAX_STEERING_ANGLE_DEGREES = 35.0 # Max steering angle for Xycar (degrees)
MAX_SPEED = 90.0
MIN_SPEED_CURVE = 40.0
MIN_SPEED_ANGLE = 50.0


def calculate_vehicle_heading_error(left_lane_fitx_px, right_lane_fitx_px, y_points_px, warped_image_height_px, ym_per_pix, xm_per_pix):
    if left_lane_fitx_px is None or right_lane_fitx_px is None or \
       left_lane_fitx_px.size == 0 or right_lane_fitx_px.size == 0 or y_points_px.size == 0:
        return 0.0, 0.0
    center_lane_x_px = (left_lane_fitx_px + right_lane_fitx_px) / 2.0
    try:
        center_lane_coeffs_meters = np.polyfit(y_points_px * ym_per_pix, center_lane_x_px * xm_per_pix, 2)
    except (np.linalg.LinAlgError, TypeError, ValueError) as e:
        rospy.logwarn_throttle(1.0, f"Polyfit failed for heading error calculation: {e}")
        return 0.0, 0.0
    y_eval_meters = warped_image_height_px * ym_per_pix
    tangent_value = 2 * center_lane_coeffs_meters[0] * y_eval_meters + center_lane_coeffs_meters[1]
    heading_error_radians = np.arctan(tangent_value)
    signed_curvature_factor_Am = center_lane_coeffs_meters[0] if len(center_lane_coeffs_meters) > 0 else 0.0
    return heading_error_radians, signed_curvature_factor_Am

def preprocess_and_get_roi(bgr_image_input):
    resized_bgr = cv2.resize(bgr_image_input, (IMAGE_RESIZED_WIDTH, IMAGE_RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)
    roi_y_start = int(IMAGE_RESIZED_HEIGHT * ROI_Y_START_PERCENTAGE)
    roi_y_end = int(IMAGE_RESIZED_HEIGHT * ROI_Y_END_PERCENTAGE)
    roi_bgr = resized_bgr[roi_y_start:roi_y_end, 0:IMAGE_RESIZED_WIDTH]
    return roi_bgr, resized_bgr

def generate_lane_masks(roi_bgr_image):
    hls_image = cv2.cvtColor(roi_bgr_image, cv2.COLOR_BGR2HLS)
    h_channel, l_channel, s_channel = hls_image[:,:,0], hls_image[:,:,1], hls_image[:,:,2]
    yellow_mask = np.zeros_like(h_channel)
    yellow_mask[(h_channel >= H_THRESHOLD_YELLOW_LANE[0]) & (h_channel <= H_THRESHOLD_YELLOW_LANE[1]) &
                (l_channel >= L_THRESHOLD_UNIVERSAL_LANE[0]) & (l_channel <= L_THRESHOLD_UNIVERSAL_LANE[1]) &
                (s_channel >= S_THRESHOLD_WHITE_YELLOW_LANE[0]) & (s_channel <= S_THRESHOLD_WHITE_YELLOW_LANE[1])] = 255
    white_mask = np.zeros_like(l_channel)
    white_mask[(l_channel >= 180) & (s_channel <= 60)] = 255
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    return combined_mask, yellow_mask, white_mask

def calculate_perspective_transform(image_to_warp, src_roi_ratios, warped_width_roi_ratio, warped_height_roi_ratio):
    img_height, img_width = image_to_warp.shape[:2]
    src_points_px = np.float32([[img_width * r[0], img_height * r[1]] for r in src_roi_ratios])
    output_warped_width = int(img_width * warped_width_roi_ratio)
    output_warped_height = int(img_height * warped_height_roi_ratio)
    dst_points_px = np.float32([[0,0], [output_warped_width-1,0], 
                                [output_warped_width-1,output_warped_height-1], [0,output_warped_height-1]])
    transform_matrix = cv2.getPerspectiveTransform(src_points_px, dst_points_px)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points_px, src_points_px)
    warped_output_image = cv2.warpPerspective(image_to_warp, transform_matrix, 
                                             (output_warped_width, output_warped_height), flags=cv2.INTER_LINEAR)
    return warped_output_image, transform_matrix, inverse_transform_matrix

def get_lane_start_points_from_histogram(binary_warped_image):
    histogram = np.sum(binary_warped_image[binary_warped_image.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base_x = np.argmax(histogram[:midpoint]) if midpoint > 0 and np.any(histogram[:midpoint]) else 0
    right_base_x = np.argmax(histogram[midpoint:]) + midpoint if midpoint < histogram.shape[0] and np.any(histogram[midpoint:]) else histogram.shape[0]-1
    return left_base_x, right_base_x

def perform_sliding_window_search(binary_warped_image, initial_leftx_base, initial_rightx_base):
    output_visualization_img = np.dstack((binary_warped_image, binary_warped_image, binary_warped_image))
    if np.max(binary_warped_image) <= 1 and binary_warped_image.ndim == 2: output_visualization_img *= 255 
    output_visualization_img = output_visualization_img.astype(np.uint8)
    window_height = np.int(binary_warped_image.shape[0] / SLIDING_WINDOW_COUNT)
    margin_px = int(binary_warped_image.shape[1] * SLIDING_WINDOW_MARGIN_PERCENTAGE)
    nonzero_pixels = binary_warped_image.nonzero()
    nonzeroy_coords, nonzerox_coords = np.array(nonzero_pixels[0]), np.array(nonzero_pixels[1])
    current_leftx, current_rightx = initial_leftx_base, initial_rightx_base
    left_lane_pixel_indices, right_lane_pixel_indices = [], []
    for window_idx in range(SLIDING_WINDOW_COUNT):
        win_y_low = binary_warped_image.shape[0] - (window_idx + 1) * window_height
        win_y_high = binary_warped_image.shape[0] - window_idx * window_height
        win_xleft_low, win_xleft_high = current_leftx - margin_px, current_leftx + margin_px
        win_xright_low, win_xright_high = current_rightx - margin_px, current_rightx + margin_px
        cv2.rectangle(output_visualization_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(output_visualization_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 
        good_left_indices = ((nonzeroy_coords >= win_y_low) & (nonzeroy_coords < win_y_high) & 
                             (nonzerox_coords >= win_xleft_low) & (nonzerox_coords < win_xleft_high)).nonzero()[0]
        good_right_indices = ((nonzeroy_coords >= win_y_low) & (nonzeroy_coords < win_y_high) & 
                              (nonzerox_coords >= win_xright_low) & (nonzerox_coords < win_xright_high)).nonzero()[0]
        left_lane_pixel_indices.append(good_left_indices)
        right_lane_pixel_indices.append(good_right_indices)
        if len(good_left_indices) > MIN_PIXELS_FOR_RECENTERING: current_leftx = np.int(np.mean(nonzerox_coords[good_left_indices]))
        if len(good_right_indices) > MIN_PIXELS_FOR_RECENTERING: current_rightx = np.int(np.mean(nonzerox_coords[good_right_indices]))
    left_lane_pixel_indices = np.concatenate(left_lane_pixel_indices) if len(left_lane_pixel_indices) > 0 and any(i.size > 0 for i in left_lane_pixel_indices) else np.array([], dtype=np.int32)
    right_lane_pixel_indices = np.concatenate(right_lane_pixel_indices) if len(right_lane_pixel_indices) > 0 and any(i.size > 0 for i in right_lane_pixel_indices) else np.array([], dtype=np.int32)
    final_leftx = nonzerox_coords[left_lane_pixel_indices] if left_lane_pixel_indices.size > 0 else np.array([])
    final_lefty = nonzeroy_coords[left_lane_pixel_indices] if left_lane_pixel_indices.size > 0 else np.array([])
    final_rightx = nonzerox_coords[right_lane_pixel_indices] if right_lane_pixel_indices.size > 0 else np.array([])
    final_righty = nonzeroy_coords[right_lane_pixel_indices] if right_lane_pixel_indices.size > 0 else np.array([])
    if len(final_leftx) > 0: output_visualization_img[final_lefty, final_leftx] = [255,0,0]
    if len(final_rightx) > 0: output_visualization_img[final_righty, final_rightx] = [0,0,255]
    return final_leftx, final_lefty, final_rightx, final_righty, output_visualization_img

def search_lane_pixels_around_poly(binary_warped_image, polyfit_left, polyfit_right):
    margin_px = int(binary_warped_image.shape[1] * POLYNOMIAL_SEARCH_MARGIN_PERCENTAGE)
    nonzero_pixels = binary_warped_image.nonzero()
    nonzeroy_coords, nonzerox_coords = np.array(nonzero_pixels[0]), np.array(nonzero_pixels[1])
    output_visualization_img = np.dstack((binary_warped_image, binary_warped_image, binary_warped_image))
    if np.max(binary_warped_image) <= 1 and binary_warped_image.ndim == 2: output_visualization_img *= 255
    output_visualization_img = output_visualization_img.astype(np.uint8)
    left_poly_line_x = polyfit_left[0]*(nonzeroy_coords**2) + polyfit_left[1]*nonzeroy_coords + polyfit_left[2]
    left_lane_indices_bool = ((nonzerox_coords > left_poly_line_x - margin_px) & (nonzerox_coords < left_poly_line_x + margin_px))
    right_poly_line_x = polyfit_right[0]*(nonzeroy_coords**2) + polyfit_right[1]*nonzeroy_coords + polyfit_right[2]
    right_lane_indices_bool = ((nonzerox_coords > right_poly_line_x - margin_px) & (nonzerox_coords < right_poly_line_x + margin_px))
    final_leftx, final_lefty = nonzerox_coords[left_lane_indices_bool], nonzeroy_coords[left_lane_indices_bool]
    final_rightx, final_righty = nonzerox_coords[right_lane_indices_bool], nonzeroy_coords[right_lane_indices_bool]
    if len(final_leftx) > 0: output_visualization_img[final_lefty, final_leftx] = [255,0,0]
    if len(final_rightx) > 0: output_visualization_img[final_righty, final_rightx] = [0,0,255]
    return final_leftx, final_lefty, final_rightx, final_righty, output_visualization_img

def apply_polynomial_fit(detected_leftx_px, detected_lefty_px, detected_rightx_px, detected_righty_px, warped_image_h_px):
    global previous_left_polynomial_fit, previous_right_polynomial_fit
    current_left_fit, current_right_fit = None, None
    # Ensure at least 3 points for a 2nd degree polynomial
    if len(detected_leftx_px) > 2 and len(detected_lefty_px) > 2:
        try: current_left_fit = np.polyfit(detected_lefty_px, detected_leftx_px, 2)
        except (np.RankWarning, Exception) as e: rospy.logwarn_throttle(1.0,f"LFitFail:{e}"); current_left_fit=previous_left_polynomial_fit
    else: current_left_fit=previous_left_polynomial_fit
    if len(detected_rightx_px) > 2 and len(detected_righty_px) > 2:
        try: current_right_fit = np.polyfit(detected_righty_px, detected_rightx_px, 2)
        except (np.RankWarning, Exception) as e: rospy.logwarn_throttle(1.0,f"RFitFail:{e}"); current_right_fit=previous_right_polynomial_fit
    else: current_right_fit=previous_right_polynomial_fit
    y_points_for_plotting = np.linspace(0,warped_image_h_px-1,warped_image_h_px)
    fitted_leftx_px, fitted_rightx_px = np.array([]), np.array([])
    if current_left_fit is not None:
        fitted_leftx_px = current_left_fit[0]*y_points_for_plotting**2+current_left_fit[1]*y_points_for_plotting+current_left_fit[2]
        previous_left_polynomial_fit = current_left_fit
    if current_right_fit is not None:
        fitted_rightx_px = current_right_fit[0]*y_points_for_plotting**2+current_right_fit[1]*y_points_for_plotting+current_right_fit[2]
        previous_right_polynomial_fit = current_right_fit
    return current_left_fit, current_right_fit, fitted_leftx_px, fitted_rightx_px, y_points_for_plotting

def get_curvature_and_offset(fitted_leftx_px, fitted_rightx_px, y_points_for_plotting, warped_image_dims_px):
    warped_h_px, warped_w_px = warped_image_dims_px
    y_eval_px = np.max(y_points_for_plotting) if y_points_for_plotting.size > 0 else warped_h_px - 1
    left_curvature_m, right_curvature_m = float('inf'), float('inf')
    if len(fitted_leftx_px)>0 and len(y_points_for_plotting)>0:
        left_fit_coeffs_meters = np.polyfit(y_points_for_plotting*YM_PER_PIX_WARPED,fitted_leftx_px*XM_PER_PIX_WARPED,2)
        if abs(left_fit_coeffs_meters[0]) > 1e-9 : left_curvature_m = ((1+(2*left_fit_coeffs_meters[0]*y_eval_px*YM_PER_PIX_WARPED+left_fit_coeffs_meters[1])**2)**1.5)/np.absolute(2*left_fit_coeffs_meters[0])
    if len(fitted_rightx_px)>0 and len(y_points_for_plotting)>0:
        right_fit_coeffs_meters = np.polyfit(y_points_for_plotting*YM_PER_PIX_WARPED,fitted_rightx_px*XM_PER_PIX_WARPED,2)
        if abs(right_fit_coeffs_meters[0]) > 1e-9 : right_curvature_m = ((1+(2*right_fit_coeffs_meters[0]*y_eval_px*YM_PER_PIX_WARPED+right_fit_coeffs_meters[1])**2)**1.5)/np.absolute(2*right_fit_coeffs_meters[0])
    lane_center_px = (fitted_leftx_px[-1]+fitted_rightx_px[-1])/2 if len(fitted_leftx_px)>0 and len(fitted_rightx_px)>0 else warped_w_px/2
    vehicle_center_px = warped_w_px/2
    offset_meters = (vehicle_center_px-lane_center_px)*XM_PER_PIX_WARPED
    return left_curvature_m,right_curvature_m,offset_meters

def draw_lane_visualization(original_bgr_image, warped_binary_image_shape, inv_perspective_matrix, 
                            fitted_leftx_px, fitted_rightx_px, y_points_for_plotting, 
                            left_curvature_m, right_curvature_m, vehicle_offset_m, final_steering_angle_deg):
    output_image = original_bgr_image.copy()
    font, font_scale, font_color, line_type = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
    text_y_start, text_y_offset = 30, 25
    if fitted_leftx_px.size==0 or fitted_rightx_px.size==0 or y_points_for_plotting.size==0:
        avg_curvature = (left_curvature_m + right_curvature_m)/2 if left_curvature_m!=float('inf') and right_curvature_m!=float('inf') else (left_curvature_m if left_curvature_m!=float('inf') else right_curvature_m)
        cv2.putText(output_image, f"Curv: {avg_curvature:.0f}m" if avg_curvature!=float('inf') else "Curv: Straight", (10,text_y_start), font,font_scale,(0,0,0),line_type)
        cv2.putText(output_image, f"Offset:{vehicle_offset_m:.2f}m(NoLines)", (10,text_y_start+text_y_offset), font,font_scale,(0,0,0),line_type)
        cv2.putText(output_image, f"Steer: {final_steering_angle_deg:.1f}deg", (10,text_y_start+2*text_y_offset), font,font_scale,(0,0,0),line_type)
        return output_image
    warp_color_overlay = np.zeros(warped_binary_image_shape,dtype=np.uint8)
    color_warp_rgb = np.dstack((warp_color_overlay,warp_color_overlay,warp_color_overlay))
    left_lane_points = np.array([np.transpose(np.vstack([fitted_leftx_px,y_points_for_plotting]))])
    right_lane_points_for_fill = np.array([np.flipud(np.transpose(np.vstack([fitted_rightx_px,y_points_for_plotting])))])
    lane_polygon_points = np.hstack((left_lane_points,right_lane_points_for_fill))
    cv2.fillPoly(color_warp_rgb,np.int_([lane_polygon_points]),(0,255,0))
    cv2.polylines(color_warp_rgb,np.int_([left_lane_points]),isClosed=False,color=(255,0,0),thickness=10)
    right_lane_points_for_line = np.array([np.transpose(np.vstack([fitted_rightx_px,y_points_for_plotting]))])
    cv2.polylines(color_warp_rgb,np.int_([right_lane_points_for_line]),isClosed=False,color=(0,0,255),thickness=10)
    unwarped_lane_overlay = cv2.warpPerspective(color_warp_rgb,inv_perspective_matrix,(original_bgr_image.shape[1],original_bgr_image.shape[0])) 
    output_image = cv2.addWeighted(output_image,1,unwarped_lane_overlay,0.3,0)
    avg_curvature = (left_curvature_m + right_curvature_m)/2 if left_curvature_m!=float('inf') and right_curvature_m!=float('inf') else (left_curvature_m if left_curvature_m!=float('inf') else right_curvature_m)
    curvature_text = f"Curv: {avg_curvature:.0f}m" if avg_curvature!=float('inf') else "Curv: Straight"
    cv2.putText(output_image,curvature_text,(10,text_y_start),font,font_scale,font_color,line_type)
    cv2.putText(output_image,f"Offset: {vehicle_offset_m:.2f}m",(10,text_y_start+text_y_offset),font,font_scale,font_color,line_type)
    cv2.putText(output_image,f"Steer Angle: {final_steering_angle_deg:.1f}deg",(10,text_y_start+2*text_y_offset),font,font_scale,font_color,line_type)
    return output_image

def main_image_processing_pipeline(cv_image):
    global previous_left_polynomial_fit, previous_right_polynomial_fit, prev_vehicle_offset_m_for_D, last_time_for_derivative_calc, accumulated_integral_offset_meters
    roi_image_bgr, resized_original_bgr = preprocess_and_get_roi(cv_image)
    combined_binary_mask, _, _ = generate_lane_masks(roi_image_bgr)
    warped_binary_mask, perspective_matrix, inverse_perspective_matrix = calculate_perspective_transform(
        combined_binary_mask, PERSPECTIVE_SRC_POINTS_ROI_RATIOS, 
        WARPED_IMAGE_WIDTH_AS_ROI_RATIO, WARPED_IMAGE_HEIGHT_AS_ROI_RATIO
    )
    warped_image_dimensions = (warped_binary_mask.shape[0], warped_binary_mask.shape[1])
    if previous_left_polynomial_fit is not None and previous_right_polynomial_fit is not None:
        detected_leftx,detected_lefty,detected_rightx,detected_righty,lane_search_visualization_img = search_lane_pixels_around_poly(warped_binary_mask,previous_left_polynomial_fit,previous_right_polynomial_fit)
        if len(detected_leftx)<MIN_PIXELS_FOR_RECENTERING or len(detected_rightx)<MIN_PIXELS_FOR_RECENTERING:
            rospy.logwarn_throttle(1.0,"SearchPolyFail->SlideWin")
            initial_leftx_base, initial_rightx_base = get_lane_start_points_from_histogram(warped_binary_mask) # Removed _ from histogram
            detected_leftx,detected_lefty,detected_rightx,detected_righty,lane_search_visualization_img = perform_sliding_window_search(warped_binary_mask,initial_leftx_base,initial_rightx_base)
    else:
        initial_leftx_base, initial_rightx_base, _ = get_lane_start_points_from_histogram(warped_binary_mask) # Keep _ if histogram not needed elsewhere
        detected_leftx,detected_lefty,detected_rightx,detected_righty,lane_search_visualization_img = perform_sliding_window_search(warped_binary_mask,initial_leftx_base,initial_rightx_base)
    current_left_fit_coeffs, current_right_fit_coeffs, fitted_leftx_plot, fitted_rightx_plot, y_points_plot = apply_polynomial_fit(detected_leftx,detected_lefty,detected_rightx,detected_righty,warped_image_dimensions[0])
    vehicle_offset_meters = 0.0; left_curvature_m,right_curvature_m = float('inf'),float('inf')
    if current_left_fit_coeffs is not None and current_right_fit_coeffs is not None and len(fitted_leftx_plot)>0 and len(fitted_rightx_plot)>0:
        left_curvature_m,right_curvature_m,vehicle_offset_meters = get_curvature_and_offset(fitted_leftx_plot,fitted_rightx_plot,y_points_plot,warped_image_dimensions)
    else: rospy.logwarn_throttle(1.0,"BadPolyFit->DefaultCurv/Offset")
    heading_error_rad,signed_curvature_factor_Am = 0.0,0.0
    if current_left_fit_coeffs is not None and current_right_fit_coeffs is not None and all(arr is not None and arr.size>0 for arr in [fitted_leftx_plot,fitted_rightx_plot,y_points_plot]):
        heading_error_rad,signed_curvature_factor_Am = calculate_vehicle_heading_error(fitted_leftx_plot,fitted_rightx_plot,y_points_plot,warped_image_dimensions[0],YM_PER_PIX_WARPED,XM_PER_PIX_WARPED)
    else: rospy.logwarn_throttle(1.0,"SkipHeadingErr->MissingFits")
    current_timestamp = rospy.get_time(); delta_time=0.0
    if last_time_for_derivative_calc>0: delta_time = current_timestamp-last_time_for_derivative_calc
    derivative_offset_term_contribution = (vehicle_offset_meters-previous_vehicle_offset_meters)/delta_time if delta_time>0.001 else 0.0
    if delta_time>0: accumulated_integral_offset_meters += vehicle_offset_meters*delta_time
    accumulated_integral_offset_meters = np.clip(accumulated_integral_offset_meters,-MAX_INTEGRAL_TERM_VALUE,MAX_INTEGRAL_TERM_VALUE)
    integral_offset_term_contribution = KI_LATERAL_OFFSET*accumulated_integral_offset_meters
    previous_vehicle_offset_meters,last_time_for_derivative_calc = vehicle_offset_meters,current_timestamp
    prop_offset_term = KP_LATERAL_OFFSET*vehicle_offset_meters
    deriv_offset_term = KD_LATERAL_OFFSET*derivative_offset_term_contribution
    prop_heading_term = KP_HEADING_ERROR*heading_error_rad
    feedforward_curvature_term = K_FF_LANE_CURVATURE*signed_curvature_factor_Am
    target_steering_angle_rad = prop_offset_term+integral_offset_term_contribution+deriv_offset_term-prop_heading_term-feedforward_curvature_term
    steering_angle_degrees = np.degrees(target_steering_angle_rad)
    final_steering_angle_degrees = np.clip(steering_angle_degrees,-MAX_STEERING_ANGLE_DEGREES,MAX_STEERING_ANGLE_DEGREES)
    avg_lane_curvature_m = float('inf')
    if left_curvature_m!=float('inf') and right_curvature_m!=float('inf'): avg_lane_curvature_m = (left_curvature_m+right_curvature_m)/2.0
    elif left_curvature_m!=float('inf'): avg_lane_curvature_m = left_curvature_m
    elif right_curvature_m!=float('inf'): avg_lane_curvature_m = right_curvature_m
    speed_based_on_curvature = MAX_SPEED
    if avg_lane_curvature_m<150: speed_based_on_curvature = MIN_SPEED_CURVE
    elif avg_lane_curvature_m<400: speed_based_on_curvature = 55.0
    elif avg_lane_curvature_m<800: speed_based_on_curvature = 70.0
    speed_based_on_angle = MAX_SPEED; abs_steering_angle = abs(final_steering_angle_degrees)
    if abs_steering_angle<5: speed_based_on_angle = MAX_SPEED
    elif abs_steering_angle<10: speed_based_on_angle = 80.0
    elif abs_steering_angle<20: speed_based_on_angle = 70.0
    else: speed_based_on_angle = MIN_SPEED_ANGLE
    final_vehicle_speed = min(speed_based_on_curvature,speed_based_on_angle)
    final_visualization_image = resized_original_bgr 
    if current_left_fit_coeffs is not None and current_right_fit_coeffs is not None and inverse_perspective_matrix is not None:
         final_visualization_image = draw_lane_visualization(resized_original_bgr,warped_binary_mask.shape,inverse_perspective_matrix,fitted_leftx_plot,fitted_rightx_plot,y_points_plot,left_curvature_m,right_curvature_m,vehicle_offset_meters,final_steering_angle_degrees)
    else:
        font,font_scale,line_type,text_y,text_off = cv2.FONT_HERSHEY_SIMPLEX,0.6,2,30,25
        cv2.putText(final_visualization_image,f"Curv:{left_curvature_m:.0f}/{right_curvature_m:.0f}(WarpErr)",(10,text_y),font,font_scale,(0,0,255),line_type)
        cv2.putText(final_visualization_image,f"Off:{vehicle_offset_meters:.2f}(WarpErr)",(10,text_y+text_off),font,font_scale,(0,0,255),line_type)
        cv2.putText(final_visualization_image,f"Steer:{final_steering_angle_degrees:.1f}(WarpErr)",(10,text_y+2*text_off),font,font_scale,(0,0,255),line_type)
    debug_data = {"vehicle_offset_m":vehicle_offset_meters,"heading_error_rad":heading_error_rad,
                  "target_steering_angle_rad":target_steering_angle_rad,"final_steering_angle_deg":final_steering_angle_degrees,
                  "speed_based_on_curvature":speed_based_on_curvature,"speed_based_on_angle":speed_based_on_angle,
                  "final_vehicle_speed":final_vehicle_speed,
                  "pid_p_offset":prop_offset_term,"pid_i_offset":integral_offset_term_contribution, 
                  "pid_d_offset":deriv_offset_term,"pid_p_heading":prop_heading_term, 
                  "ff_curvature":feedforward_curvature_term,
                  "left_curvature_m":left_curvature_m,"right_curvature_m":right_curvature_m,
                  "accumulated_integral_offset_m":accumulated_integral_offset_meters}
    return final_steering_angle_degrees,final_vehicle_speed,final_visualization_image,combined_binary_mask,warped_binary_mask,lane_search_visualization_img,debug_data

def ros_image_callback(ros_img_msg):
    global current_image_cv
    current_image_cv = bridge.imgmsg_to_cv2(ros_img_msg,"bgr8")

def publish_motor_commands(steering_angle,vehicle_speed):
    if motor_control_publisher is not None:
        motor_msg = XycarMotor(); motor_msg.angle=float(steering_angle); motor_msg.speed=float(vehicle_speed)
        motor_control_publisher.publish(motor_msg)

def ros_main_loop():
    global current_image_cv,motor_control_publisher,previous_left_polynomial_fit,previous_right_polynomial_fit, \
           last_time_for_derivative_calc, prev_vehicle_offset_m_for_D, accumulated_integral_offset_meters
    rospy.init_node('advanced_lane_keeping_controller')
    motor_control_publisher = rospy.Publisher('/xycar_motor',XycarMotor,queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw',Image,ros_image_callback,queue_size=1)
    rospy.loginfo("Waiting for initial image message...")
    while current_image_cv is None and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0,"Still no image received...")
        time.sleep(0.1)
    rospy.loginfo("Initial image received. Starting Advanced Lane Keeping Controller.")
    print("▶▶▶ Advanced Lane Detection & Control Algorithm Initialized")
    last_time_for_derivative_calc = rospy.get_time()
    prev_vehicle_offset_m_for_D = 0.0
    accumulated_integral_offset_meters = 0.0
    loop_rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if current_image_cv is None:
            loop_rate.sleep(); continue
        local_image_copy = current_image_cv.copy()
        final_angle,final_speed,viz_final,viz_mask,viz_warped,viz_search,dbg_info = main_image_processing_pipeline(local_image_copy)
        publish_motor_commands(final_angle,final_speed)
        log_output_str = (f"Off:{dbg_info['vehicle_offset_m']:.2f}m, HeadErr:{np.degrees(dbg_info.get('heading_error_rad',0)):.1f}d, "
                   f"RawSteer:{np.degrees(dbg_info.get('target_steering_angle_rad',0)):.1f}d, FinalSteer:{final_angle:.1f}d, "
                   f"SpeedCrv:{dbg_info.get('speed_based_on_curvature',0):.0f}, SpeedAng:{dbg_info.get('speed_based_on_angle',0):.0f}, FinalSpeed:{final_speed:.0f} | "
                   f"P:{dbg_info.get('pid_p_offset',0):.1f},I:{dbg_info.get('pid_i_offset',0):.1f},D:{dbg_info.get('pid_d_offset',0):.1f},H:{dbg_info.get('pid_p_heading',0):.1f},FF:{dbg_info.get('ff_curvature',0):.1f}")
        rospy.loginfo(log_output_str)
        cv2.imshow("Final Lane Visualization",viz_final)
        cv2.imshow("Combined Binary Mask (from ROI)",viz_mask)
        cv2.imshow("Warped Binary Mask",viz_warped)
        cv2.imshow("Lane Pixel Search Detail",viz_search)
        if cv2.waitKey(1)&0xFF == ord('q'):
            rospy.loginfo("Shutdown key (q) pressed. Exiting.")
            break
        loop_rate.sleep()
if __name__ == '__main__':
    try: ros_main_loop()
    except rospy.ROSInterruptException: rospy.loginfo("ROSInterruptException caught during shutdown. Node exiting.")
    except Exception as e: rospy.logerr(f"Unhandled critical exception in main: {e}")
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("OpenCV windows closed. Xycar controller node has shut down.")
