#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge

# Global variables for ROS
bridge = CvBridge()
current_image = None
motor_publisher = None

# Global variables for lane detection state
previous_left_fit = None
previous_right_fit = None
prev_vehicle_offset_m_for_D = 0.0 # For D term of PID
last_time_for_D = 0.0             # For D term of PID

# PID and Feed-Forward Gains (NEEDS CAREFUL TUNING)
KP_OFFSET = 30.0      # Proportional gain for vehicle offset (center_diff)
KD_OFFSET = 10.0      # Derivative gain for vehicle offset
KP_HEADING = 25.0     # Proportional gain for heading error
K_FF_CURVATURE = 0.8  # Feed-forward gain for lane curvature (signed_curvature_factor)


# Constants for image processing and lane detection
ROI_Y_START_RATIO = 0.55
ROI_Y_END_RATIO = 0.95
RESIZED_WIDTH = 320
RESIZED_HEIGHT = 240
H_THRESH_YELLOW = (15, 35)
L_THRESH_UNIVERSAL = (30, 200)
S_THRESH_WHITE_YELLOW = (100, 255)
SRC_POINTS_ROI_RATIOS = np.float32([
    [0.40, 0.10], [0.60, 0.10],
    [0.95, 0.90], [0.05, 0.90]
])
WARPED_WIDTH_RATIO = 1.0
WARPED_HEIGHT_RATIO = 1.5
N_WINDOWS = 10
WINDOW_MARGIN_PERCENT = 0.15
MIN_PIXELS_RECENTER = 50
POLY_SEARCH_MARGIN_PERCENT = 0.10
YM_PER_PIX = 30.0 / 720.0 
XM_PER_PIX = 3.7 / 700.0

def calculate_heading_error(left_fitx_px, right_fitx_px, ploty_px, H_warped_px, current_ym_per_pix, current_xm_per_pix):
    if left_fitx_px is None or right_fitx_px is None or left_fitx_px.size == 0 or right_fitx_px.size == 0 or ploty_px.size == 0:
        return 0.0, 0.0 
    center_x_px = (left_fitx_px + right_fitx_px) / 2.0
    try:
        center_fit_coeffs_m = np.polyfit(ploty_px * current_ym_per_pix, center_x_px * current_xm_per_pix, 2)
    except (np.linalg.LinAlgError, TypeError, ValueError) as e:
        rospy.logwarn_throttle(1.0, f"Polyfit failed for heading error calculation: {e}")
        return 0.0, 0.0
    y_eval_m = H_warped_px * current_ym_per_pix
    tangent_val = 2 * center_fit_coeffs_m[0] * y_eval_m + center_fit_coeffs_m[1]
    heading_error_rad = np.arctan(tangent_val)
    signed_curvature_factor = center_fit_coeffs_m[0] if len(center_fit_coeffs_m) > 0 else 0.0
    return heading_error_rad, signed_curvature_factor
    
def preprocess_image(bgr_image):
    resized_bgr = cv2.resize(bgr_image, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)
    roi_y_start = int(RESIZED_HEIGHT * ROI_Y_START_RATIO)
    roi_y_end = int(RESIZED_HEIGHT * ROI_Y_END_RATIO)
    roi_x_start = 0
    roi_x_end = RESIZED_WIDTH
    roi_bgr = resized_bgr[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    return roi_bgr, resized_bgr

def create_masks(roi_bgr):
    hls_image = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
    h_channel, l_channel, s_channel = hls_image[:,:,0], hls_image[:,:,1], hls_image[:,:,2]
    yellow_mask = np.zeros_like(h_channel)
    yellow_mask[(h_channel >= H_THRESH_YELLOW[0]) & (h_channel <= H_THRESH_YELLOW[1]) &
                (l_channel >= L_THRESH_UNIVERSAL[0]) & (l_channel <= L_THRESH_UNIVERSAL[1]) &
                (s_channel >= S_THRESH_WHITE_YELLOW[0]) & (s_channel <= S_THRESH_WHITE_YELLOW[1])] = 255
    refined_white_mask = np.zeros_like(l_channel)
    refined_white_mask[(l_channel >= 180) & (s_channel <= 60)] = 255
    combined_mask = cv2.bitwise_or(yellow_mask, refined_white_mask)
    return combined_mask, yellow_mask, refined_white_mask

def perspective_transform(img, src_points_ratios, dst_width_ratio, dst_height_ratio):
    height, width = img.shape[:2]
    src_points = np.float32([[width*r[0], height*r[1]] for r in src_points_ratios])
    warped_width = int(width * dst_width_ratio)
    warped_height = int(height * dst_height_ratio)
    dst_points = np.float32([[0,0],[warped_width-1,0],[warped_width-1,warped_height-1],[0,warped_height-1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_img = cv2.warpPerspective(img, matrix, (warped_width, warped_height), flags=cv2.INTER_LINEAR)
    return warped_img, matrix, inv_matrix

def find_lane_starting_points(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint]) if midpoint > 0 and np.any(histogram[:midpoint]) else 0
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint if midpoint < histogram.shape[0] and np.any(histogram[midpoint:]) else histogram.shape[0]-1
    return leftx_base, rightx_base, histogram

def sliding_window_search(binary_warped, leftx_base, rightx_base):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    if np.max(binary_warped) <= 1 and binary_warped.ndim == 2: out_img *= 255 
    out_img = out_img.astype(np.uint8)
    nwindows = N_WINDOWS
    window_height = np.int(binary_warped.shape[0]/nwindows)
    margin = int(binary_warped.shape[1] * WINDOW_MARGIN_PERCENT)
    minpix = MIN_PIXELS_RECENTER
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    for window in range(nwindows):
        win_y_low, win_y_high = binary_warped.shape[0]-(window+1)*window_height, binary_warped.shape[0]-window*window_height
        win_xleft_low, win_xleft_high = leftx_current-margin, leftx_current+margin
        win_xright_low, win_xright_high = rightx_current-margin, rightx_current+margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2) 
        good_left_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds); right_lane_inds.append(good_right_inds)
        if len(good_left_inds)>minpix: leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix: rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds)>0 and any(i.size > 0 for i in left_lane_inds) else np.array([], dtype=np.int32)
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds)>0 and any(i.size > 0 for i in right_lane_inds) else np.array([], dtype=np.int32)
    leftx = nonzerox[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size > 0 else np.array([])
    if len(leftx)>0: out_img[lefty,leftx] = [255,0,0]
    if len(rightx)>0: out_img[righty,rightx] = [0,0,255]
    return leftx,lefty,rightx,righty,out_img

def search_around_previous_fit(binary_warped, left_fit, right_fit):
    margin = int(binary_warped.shape[1] * POLY_SEARCH_MARGIN_PERCENT)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))
    if np.max(binary_warped) <= 1 and binary_warped.ndim == 2: out_img *= 255
    out_img = out_img.astype(np.uint8)
    left_lane_inds_bool = ((nonzerox > (left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy+left_fit[2]-margin)) & 
                           (nonzerox < (left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy+left_fit[2]+margin))) 
    right_lane_inds_bool = ((nonzerox > (right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy+right_fit[2]-margin)) & 
                            (nonzerox < (right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy+right_fit[2]+margin)))
    left_lane_inds = left_lane_inds_bool.nonzero()[0]
    right_lane_inds = right_lane_inds_bool.nonzero()[0]
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    if len(leftx)>0: out_img[lefty,leftx] = [255,0,0]
    if len(rightx)>0: out_img[righty,rightx] = [0,0,255]
    return leftx,lefty,rightx,righty,out_img

def fit_polynomial(leftx, lefty, rightx, righty, warped_height):
    global previous_left_fit, previous_right_fit
    left_fit, right_fit = None, None
    if len(leftx)>0 and len(lefty)>0:
        try: left_fit = np.polyfit(lefty,leftx,2)
        except (np.RankWarning, Exception) as e: rospy.logwarn_throttle(1.0,f"LFitFail:{e}"); left_fit=previous_left_fit
    else: left_fit=previous_left_fit
    if len(rightx)>0 and len(righty)>0:
        try: right_fit = np.polyfit(righty,rightx,2)
        except (np.RankWarning, Exception) as e: rospy.logwarn_throttle(1.0,f"RFitFail:{e}"); right_fit=previous_right_fit
    else: right_fit=previous_right_fit
    ploty = np.linspace(0,warped_height-1,warped_height)
    left_fitx, right_fitx = np.array([]), np.array([])
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        previous_left_fit = left_fit
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
        previous_right_fit = right_fit
    return left_fit,right_fit,left_fitx,right_fitx,ploty

def calculate_curvature_offset(left_fitx, right_fitx, ploty, warped_dims):
    warped_height, warped_width = warped_dims
    y_eval = np.max(ploty) if ploty.size > 0 else warped_height -1
    left_curverad, right_curverad = float('inf'), float('inf')
    if len(left_fitx)>0 and len(ploty)>0:
        left_fit_cr = np.polyfit(ploty*YM_PER_PIX,left_fitx*XM_PER_PIX,2)
        if abs(left_fit_cr[0]) > 1e-9 : left_curverad = ((1+(2*left_fit_cr[0]*y_eval*YM_PER_PIX+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    if len(right_fitx)>0 and len(ploty)>0:
        right_fit_cr = np.polyfit(ploty*YM_PER_PIX,right_fitx*XM_PER_PIX,2)
        if abs(right_fit_cr[0]) > 1e-9 : right_curverad = ((1+(2*right_fit_cr[0]*y_eval*YM_PER_PIX+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
    lane_center_px = (left_fitx[-1]+right_fitx[-1])/2 if len(left_fitx)>0 and len(right_fitx)>0 else warped_width/2
    car_center_px = warped_width/2
    offset_m = (car_center_px-lane_center_px)*XM_PER_PIX
    return left_curverad,right_curverad,offset_m

def visualize_lanes(original_img, warped_binary_shape, inv_matrix, left_fitx, right_fitx, ploty, 
                    left_cur, right_cur, offset, final_steering_angle): # Added final_steering_angle
    final_image = original_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255,255,255) # White
    line_type = 2
    text_y_start = 30
    text_y_offset = 25

    if left_fitx.size==0 or right_fitx.size==0 or ploty.size==0:
        avg_cur_display = (left_cur + right_cur) / 2.0 if left_cur!=float('inf') and right_cur!=float('inf') else (left_cur if left_cur!=float('inf') else right_cur)
        cv2.putText(final_image,f"Curvature: {avg_cur_display:.0f}m" if avg_cur_display!=float('inf') else "Curvature: Straight",(10,text_y_start),font,font_scale,(0,0,0),line_type)
        cv2.putText(final_image,f"Offset:{offset:.2f}m(NoLines)",(10,text_y_start+text_y_offset),font,font_scale,(0,0,0),line_type)
        cv2.putText(final_image,f"Steer Angle:{final_steering_angle:.1f}deg",(10,text_y_start+2*text_y_offset),font,font_scale,(0,0,0),line_type)
        return final_image
        
    warp_zero = np.zeros(warped_binary_shape,dtype=np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right_fill = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right_fill))
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
    cv2.polylines(color_warp,np.int_([pts_left]),isClosed=False,color=(255,0,0),thickness=10) # Blue for left
    pts_right_line = np.array([np.transpose(np.vstack([right_fitx,ploty]))])
    cv2.polylines(color_warp,np.int_([pts_right_line]),isClosed=False,color=(0,0,255),thickness=10) # Red for right
    
    newwarp = cv2.warpPerspective(color_warp,inv_matrix,(original_img.shape[1],original_img.shape[0])) 
    final_image = cv2.addWeighted(final_image,1,newwarp,0.3,0)
    
    avg_cur_display = (left_cur + right_cur) / 2.0 if left_cur!=float('inf') and right_cur!=float('inf') else (left_cur if left_cur!=float('inf') else right_cur)
    cur_text = f"Curvature: {avg_cur_display:.0f}m" if avg_cur_display!=float('inf') else "Curvature: Straight"
    cv2.putText(final_image,cur_text,(10,text_y_start),font,font_scale,font_color,line_type)
    cv2.putText(final_image,f"Offset: {offset:.2f}m",(10,text_y_start+text_y_offset),font,font_scale,font_color,line_type)
    cv2.putText(final_image,f"Steer Angle: {final_steering_angle:.1f} deg",(10,text_y_start+2*text_y_offset),font,font_scale,font_color,line_type)
    return final_image

def process_image(image_input):
    global previous_left_fit, previous_right_fit, prev_vehicle_offset_m_for_D, last_time_for_D
    roi_bgr, resized_bgr_for_display = preprocess_image(image_input)
    combined_mask, yellow_mask, white_mask = create_masks(roi_bgr)
    warped_mask, pers_matrix, inv_pers_matrix = perspective_transform(combined_mask, SRC_POINTS_ROI_RATIOS, WARPED_WIDTH_RATIO, WARPED_HEIGHT_RATIO)
    warped_dims = (warped_mask.shape[0], warped_mask.shape[1])
    if previous_left_fit is not None and previous_right_fit is not None:
        leftx,lefty,rightx,righty,lane_search_viz = search_around_previous_fit(warped_mask,previous_left_fit,previous_right_fit)
        if len(leftx)<MIN_PIXELS_RECENTER or len(rightx)<MIN_PIXELS_RECENTER:
            rospy.logwarn_throttle(1.0,"SearchPolyFail->SlideWin")
            left_base,right_base,_ = find_lane_starting_points(warped_mask)
            leftx,lefty,rightx,righty,lane_search_viz = sliding_window_search(warped_mask,left_base,right_base)
    else:
        left_base,right_base,_ = find_lane_starting_points(warped_mask)
        leftx,lefty,rightx,righty,lane_search_viz = sliding_window_search(warped_mask,left_base,right_base)
    left_fit,right_fit,left_fitx,right_fitx,ploty = fit_polynomial(leftx,lefty,rightx,righty,warped_dims[0])
    center_diff = 0.0; left_cur,right_cur = float('inf'),float('inf')
    if left_fit is not None and right_fit is not None and len(left_fitx)>0 and len(right_fitx)>0:
        left_cur,right_cur,center_diff = calculate_curvature_offset(left_fitx,right_fitx,ploty,warped_dims)
    else: rospy.logwarn_throttle(1.0,"BadPolyFit->DefaultCurv/Offset")
    heading_error_rad,signed_curvature_factor = 0.0,0.0
    if left_fit is not None and right_fit is not None and all(f is not None for f in [left_fitx,right_fitx,ploty]) and all(f.size>0 for f in [left_fitx,right_fitx,ploty]):
        heading_error_rad,signed_curvature_factor = calculate_heading_error(left_fitx,right_fitx,ploty,warped_dims[0],YM_PER_PIX,XM_PER_PIX)
    else: rospy.logwarn_throttle(1.0,"SkipHeadingErr->MissingFits")
    
    current_time = rospy.get_time(); dt=0.0
    if last_time_for_D>0: dt = current_time-last_time_for_D
    d_offset_dt = (center_diff-prev_vehicle_offset_m_for_D)/dt if dt>0.001 else 0.0
    prev_vehicle_offset_m_for_D,last_time_for_D = center_diff,current_time
    p_offset_term = KP_OFFSET*center_diff
    d_offset_term = KD_OFFSET*d_offset_dt
    p_heading_term = KP_HEADING*heading_error_rad
    ff_curvature_term = K_FF_CURVATURE*signed_curvature_factor
    target_angle_rad = p_offset_term - p_heading_term - ff_curvature_term + d_offset_term
    angle_deg = np.degrees(target_angle_rad)
    angle = np.clip(angle_deg,-35.0,35.0)
    
    avg_curvature = float('inf')
    if left_cur != float('inf') and right_cur != float('inf'): avg_curvature = (left_cur + right_cur) / 2.0
    elif left_cur != float('inf'): avg_curvature = left_cur
    elif right_cur != float('inf'): avg_curvature = right_cur

    base_speed_for_curve = 90.0
    if avg_curvature < 150: base_speed_for_curve = 40.0
    elif avg_curvature < 400: base_speed_for_curve = 55.0
    elif avg_curvature < 800: base_speed_for_curve = 70.0
    
    speed_due_to_angle = 90.0; abs_final_angle = abs(angle)
    if abs_final_angle < 5: speed_due_to_angle = 90.0
    elif abs_final_angle < 10: speed_due_to_angle = 80.0
    elif abs_final_angle < 20: speed_due_to_angle = 70.0
    else: speed_due_to_angle = 50.0
            
    speed = min(base_speed_for_curve,speed_due_to_angle)

    final_visualization = resized_bgr_for_display # Default if visualization fails
    if left_fit is not None and right_fit is not None and inv_pers_matrix is not None:
         final_visualization = visualize_lanes(resized_bgr_for_display, warped_mask.shape, inv_pers_matrix, 
                                               left_fitx, right_fitx, ploty, 
                                               left_cur, right_cur, center_diff, angle) # Pass angle here
    else: # Fallback if Minv is not available or fits failed for drawing lane area
        final_visualization = resized_bgr_for_display.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6; line_type = 2; text_y_start = 30; text_y_offset = 25
        cv2.putText(final_visualization, f"Curvature: {left_cur:.0f}m R:{right_cur:.0f}m (Warp/Fit Error)", 
                    (10, text_y_start), font, font_scale, (0,0,255), line_type) # Red text
        cv2.putText(final_visualization, f"Offset: {center_diff:.2f}m (Warp/Fit Error)", 
                    (10, text_y_start + text_y_offset), font, font_scale, (0,0,255), line_type)
        cv2.putText(final_visualization, f"Steer Angle: {angle:.1f} deg (Warp/Fit Error)", 
                    (10, text_y_start + 2*text_y_offset), font, font_scale, (0,0,255), line_type)

    debug_info = {"left_curvature":left_cur,"right_curvature":right_cur,"center_diff":center_diff,
                  "heading_error_rad":heading_error_rad,"signed_curvature_factor":signed_curvature_factor,
                  "pid_p_offset":p_offset_term,"pid_d_offset":d_offset_term,"pid_p_heading":p_heading_term,
                  "ff_curvature":ff_curvature_term,"target_angle_rad":target_angle_rad,"final_angle_deg":angle,
                  "avg_curvature":avg_curvature, "base_speed_for_curve":base_speed_for_curve,
                  "speed_due_to_angle":speed_due_to_angle, "final_speed":speed}
    return angle,speed,final_visualization,combined_mask,warped_mask,lane_search_viz,debug_info

def image_callback(img_msg):
    global current_image
    current_image = bridge.imgmsg_to_cv2(img_msg,"bgr8")

def drive_car(angle,speed):
    if motor_publisher is not None:
        msg = XycarMotor(); msg.angle=float(angle); msg.speed=float(speed)
        motor_publisher.publish(msg)

def main_loop():
    global current_image,motor_publisher,previous_left_fit,previous_right_fit,last_time_for_D,prev_vehicle_offset_m_for_D
    rospy.init_node('advanced_lane_keeping_node')
    motor_publisher = rospy.Publisher('/xycar_motor',XycarMotor,queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw',Image,image_callback,queue_size=1)
    rospy.loginfo("Waiting for image topics...")
    while current_image is None and not rospy.is_shutdown():
        rospy.loginfo_throttle(1.0,"No image received yet...")
        time.sleep(0.1)
    rospy.loginfo("Image received. Starting advanced lane keeping node.")
    print("▶▶▶ Advanced Lane Detection Algorithm Start")
    last_time_for_D = rospy.get_time()
    prev_vehicle_offset_m_for_D = 0.0
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if current_image is None:
            rate.sleep(); continue
        img_to_process = current_image.copy()
        angle,speed,final_viz,combined_mask_viz,warped_viz,lane_search_viz,debug_info = process_image(img_to_process)
        drive_car(angle,speed)
        log_msg = (f"Off:{debug_info['center_diff']:.2f}m, HeadErr:{np.degrees(debug_info.get('heading_error_rad',0)):.1f}d, "
                   f"RawSteer:{np.degrees(debug_info.get('target_angle_rad',0)):.1f}d, FinalSteer:{angle:.1f}d, "
                   f"SpeedCrv:{debug_info.get('base_speed_for_curve',0):.0f}, SpeedAng:{debug_info.get('speed_due_to_angle',0):.0f}, FinalSpeed:{speed:.0f} "
                   f"P:{debug_info.get('pid_p_offset',0):.1f},D:{debug_info.get('pid_d_offset',0):.1f},H:{debug_info.get('pid_p_heading',0):.1f},FF:{debug_info.get('ff_curvature',0):.1f}")
        rospy.loginfo(log_msg)
        cv2.imshow("Final Visualization",final_viz)
        cv2.imshow("Combined Binary Mask (ROI)",combined_mask_viz)
        cv2.imshow("Warped Mask",warped_viz)
        cv2.imshow("Lane Search Visualization",lane_search_viz)
        if cv2.waitKey(1)&0xFF == ord('q'): break
        rate.sleep()
if __name__ == '__main__':
    try: main_loop()
    except rospy.ROSInterruptException: rospy.loginfo("ROSInterruptException caught. Shutting down.")
    except Exception as e: rospy.logerr(f"Unhandled exception in main_loop: {e}")
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("OpenCV windows closed. Node exiting.")
