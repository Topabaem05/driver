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

# Helper functions (if any were part of the new script, they'd be here)
# For this script, all logic is within start() or global scope

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def drive(angle, speed):
    msg = XycarMotor()
    msg.angle = float(angle)
    msg.speed = float(speed)
    motor_pub.publish(msg)

def start():
    global motor_pub, frame_count, prev_angle, white_lost_count # image is already global

    rospy.init_node('xycar_lane_keeper') # Or a new name like 'xycar_lane_keeper'
    motor_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    
    rospy.loginfo("Waiting for image topics...")
    rospy.wait_for_message("/usb_cam/image_raw", Image) # Wait until the first image is received
    rospy.loginfo("Image received. Starting lane keeping node.")
    
    # Original code had rospy.sleep(1.0) here, wait_for_message is more robust

    print("▶▶▶ Reactive Fast Driving Mode Start (Lane Keeping Only)")

    while not rospy.is_shutdown():
        if image is None:
            # This check might be redundant now due to wait_for_message, 
            # but good for safety if subscription is lost.
            rospy.logwarn_throttle(1.0, "Image is None, skipping frame.")
            time.sleep(0.1) # Prevent busy-looping if image is consistently None
            continue

        # Define height and width here, as image is guaranteed not to be None
        height, width = image.shape[:2]
        
        # ROI selection
        # The exact ROI might need tuning for different Xycar camera setups
        roi_top_ratio = 0.4 # Percentage of height from the top to discard
        roi = image[int(height * roi_top_ratio):, :]
        roi_h, roi_w = roi.shape[:2] # Height and width of the ROI

        # HSV conversion
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # White line detection
        lower_white = np.array([0, 0, 160])    # Adjust these values based on lighting
        upper_white = np.array([180, 50, 255]) # Adjust these values based on lighting
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Yellow line detection
        lower_yellow = np.array([15, 80, 80])   # Adjust these values
        upper_yellow = np.array([35, 255, 255]) # Adjust these values
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        frame_count += 1
        
        # Initialize angle, speed, direction for this frame
        current_angle = 0.0
        current_speed = 60.0 # Default speed
        current_direction = "INIT"

        if frame_count <= 50: # Initial straight driving period
            current_angle = 0.0
            current_speed = 60.0
            current_direction = "INIT STRAIGHT"
        else:
            total_white = cv2.countNonZero(white_mask)

            if total_white > 300: # Threshold for considering white lines detected
                white_lost_count = 0

                # Use roi_w for calculations as masks are from ROI
                left_mask_roi  = white_mask[:, :roi_w//3]
                right_mask_roi = white_mask[:, 2*roi_w//3:]
                
                left_ratio = 0.0
                if left_mask_roi.size > 0:
                    left_ratio  = cv2.countNonZero(left_mask_roi) / float(left_mask_roi.size)

                right_ratio = 0.0
                if right_mask_roi.size > 0:
                    right_ratio = cv2.countNonZero(right_mask_roi) / float(right_mask_roi.size)

                error = (left_ratio - right_ratio) * 100 # Percentage difference
                
                # P-controller for steering, Kp = 0.9
                # Max angle is 35 degrees
                current_angle = np.clip(error * 0.9, -35.0, 35.0) 
                current_direction = "WHITE TRACK"

                white_nonzero = cv2.findNonZero(white_mask)
                if white_nonzero is not None:
                    x_coords = white_nonzero[:, 0, 0]
                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    
                    track_center_pixel = (min_x + max_x) // 2
                    roi_center_pixel = roi_w // 2
                    
                    center_offset_pixel = roi_center_pixel - track_center_pixel
                    
                    # Additional P-controller for centering based on offset, Kp = 0.1
                    current_angle += center_offset_pixel * 0.1
                    current_angle = np.clip(current_angle, -50.0, 50.0) # Clip again if needed, original was -35, 35

                    # Check if track width is too narrow (possible sign of being outside lines)
                    if (max_x - min_x) < (roi_w * 0.35): # 35% of ROI width
                        current_direction += " | OUTSIDE WARNING"
                        current_speed = 35.0 # Reduce speed
            else: # White lines lost or insufficient
                white_lost_count += 1
                # Try to use yellow line as fallback
                M = cv2.moments(yellow_mask)
                if M['m00'] > 0: # If yellow line detected
                    cx = int(M['m10'] / M['m00'])
                    roi_center_pixel = roi_w // 2
                    error_pixel = cx - roi_center_pixel
                    
                    # P-controller for yellow line centering
                    # Original factor 0.005 was very small, potentially an error.
                    # Let's use a slightly larger factor, e.g., 0.05, still needs tuning.
                    # Max angle 25 degrees
                    current_angle = np.clip(error_pixel * 0.05, -25.0, 25.0) 
                    current_direction = "YELLOW FALLBACK"
                else: # No lines detected
                    current_angle = 0.0 # Or perhaps prev_angle to maintain last known good steering?
                    current_direction = "NO LINE"
                current_speed = 45.0 # Speed when on yellow or no lines

        # Steering angle smoothing
        max_delta = 3.0  # Max change in angle per frame (degrees)
        if abs(current_angle - prev_angle) > max_delta:
            current_angle = prev_angle + np.sign(current_angle - prev_angle) * max_delta
        prev_angle = current_angle # Update prev_angle with the smoothed angle

        # Final speed adjustments based on smoothed steering angle
        # These apply on top of speeds set by line detection logic
        abs_final_angle = abs(current_angle)
        if abs_final_angle < 5: # Driving relatively straight
            current_speed = max(current_speed, 90.0)
        elif abs_final_angle < 10:
            current_speed = max(current_speed, 80.0)
        elif abs_final_angle < 20:
            current_speed = max(current_speed, 70.0)
        else: # Sharper turns
            current_speed = max(current_speed, 60.0)

        if "NO LINE" in current_direction: # If lines are lost, reduce speed significantly
            current_speed = 30.0

        # Print current status
        # Using rospy.loginfo for ROS-idiomatic logging
        rospy.loginfo(f"[INFO] Mode: {current_direction}, Angle: {current_angle:.2f}, Speed: {current_speed:.2f}")

        # Display images
        cv2.imshow("Original Image", image)
        cv2.imshow("ROI", roi)
        cv2.imshow("White Mask", white_mask)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.waitKey(1) # Necessary for OpenCV windows to update

        # Send motor command
        drive(current_angle, current_speed)
        
        # Control loop rate
        time.sleep(0.02) # 50 Hz, adjust as needed

if __name__ == '__main__':
    try:
        start()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Cleanup OpenCV windows on exit
        cv2.destroyAllWindows()
