#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
# Note: time module is not explicitly used in the provided snippets,
# but if time.sleep(0.02) is needed, it should be imported:
# import time

# --------------------------------------------------------------------
# Functions from collision_avoidance_helpers.py
# --------------------------------------------------------------------

def detect_cones_sam(image):
    # TODO: MOCK SAM IMPLEMENTATION
    # This function currently returns predefined mock cone data.
    # For a real system:
    # 1. Integrate with the Segment Anything Model (SAM).
    # 2. Use SAM's AutomaticMaskGenerator or process its output to find cones.
    # 3. Convert SAM outputs (masks) into bounding boxes and centers of mass (in pixel coordinates).
    # 4. Return a list of dictionaries, each representing a detected cone,
    #    e.g., {'bbox': [x_min, y_min, x_max, y_max], 'center_of_mass': [cx, cy], 'mask': sam_mask}.
    """
    Detects cones in an image using a mock SAM function.
    
    Args:
        image: A CV2 image (e.g., np.ndarray).
        
    Returns:
        A list of dictionaries, where each dictionary represents a detected cone
        and has keys like 'bbox': [x_min, y_min, x_max, y_max] and 
        'center_of_mass': [cx, cy].
    """
    # Mock SAM function
    # Example return: [{'bbox': [100, 200, 150, 300], 'center_of_mass': [125, 250]}]
    # Returns an empty list if no cones are "detected".
    return [{'bbox': [100, 200, 150, 300], 'center_of_mass': [125, 250]}]

def get_cone_distance_and_vehicle_pose():
    # TODO: MOCK SENSOR DATA AND COORDINATE TRANSFORMATION
    # This function currently returns predefined mock vehicle pose and a single obstacle's world position.
    # For a real system:
    # 1. `pos_obstacle_world`: This should be calculated for EACH detected cone.
    #    - Input: Cone's 2D pixel coordinates (from SAM output from detect_cones_sam).
    #    - Obtain depth to the cone (e.g., from LiDAR, Stereo Camera, or Monocular Depth Estimation model).
    #    - Convert 2D pixel + depth to a 3D point in camera coordinates.
    #    - Transform the 3D camera point to 3D world coordinates using the vehicle's current pose (from sensors).
    # 2. `pos_vehicle`, `vel_vehicle`: These should be obtained from actual vehicle sensors
    #    (e.g., IMU, GPS, wheel odometry, SLAM/localization algorithms).
    """
    Returns mock sensor data for cone and vehicle pose.
    Returns mock world position for ONE obstacle, and vehicle position/velocity
    
    Returns:
        A tuple containing:
            pos_obstacle_world: A NumPy array for obstacle position in world coordinates.
            pos_vehicle: A NumPy array for vehicle position.
            vel_vehicle: A NumPy array for vehicle velocity.
    """
    # Mock sensor data function
    # Returns mock world position for ONE obstacle, and vehicle position/velocity
    pos_obstacle_world = np.array([7.0, 0.3])  # Example: Cone 7m ahead, 0.3m to the right
    pos_vehicle = np.array([0.0, 0.0])       # Vehicle at origin
    vel_vehicle = np.array([5.0, 0.0])       # Vehicle moving forward at 5 m/s
    return pos_obstacle_world, pos_vehicle, vel_vehicle

def calculate_collision_cone_avoidance(pos_obstacle, pos_vehicle, vel_vehicle, vehicle_radius, cone_radius, margin):
    """
    Calculates the avoidance angle using collision cone logic.
    
    Args:
        pos_obstacle: NumPy array for obstacle position.
        pos_vehicle: NumPy array for vehicle position.
        vel_vehicle: NumPy array for vehicle velocity.
        vehicle_radius: Float, radius of the vehicle.
        cone_radius: Float, radius of the cone.
        margin: Float, additional safety margin.
        
    Returns:
        A tuple containing:
            phi: Avoidance angle in radians.
            avoidance_active: Boolean, True if avoidance is active, False otherwise.
    """
    r = pos_obstacle - pos_vehicle
    
    if np.linalg.norm(r) == 0:
        return 0.0, False
        
    r_safe = vehicle_radius + cone_radius + margin
    
    # Using arctan2 is often more stable than arctan for ratios
    theta = np.arctan2(r_safe, np.linalg.norm(r))
    
    if np.linalg.norm(vel_vehicle) == 0 or np.linalg.norm(r) == 0:
        return 0.0, False
        
    # Handle potential ValueError from np.arccos if the argument is outside [-1, 1]
    # due to floating point inaccuracies by clipping the value.
    dot_product_normalized = np.dot(vel_vehicle, r) / (np.linalg.norm(vel_vehicle) * np.linalg.norm(r))
    alpha = np.arccos(np.clip(dot_product_normalized, -1.0, 1.0))
    
    if alpha < theta:
        # np.cross will be a scalar for 2D vectors
        phi = np.sign(np.cross(vel_vehicle, r)) * (theta - alpha)
        return phi, True
    else:
        return 0.0, False

# --------------------------------------------------------------------
# Main script logic from xycar_drive_with_avoidance.py
# --------------------------------------------------------------------

image = np.empty(shape=[0])
bridge = CvBridge()
motor = None # Will be initialized as Publisher in start()

# Global constants for collision avoidance and general driving
VEHICLE_RADIUS = 0.2  # Example value in meters
CONE_RADIUS = 0.1     # Example value in meters
SAFETY_MARGIN = 0.3   # Example value in meters

CAM_FPS = 30
WIDTH, HEIGHT = 640, 480

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def drive(Angle, Speed): 
    global motor # motor is the Publisher object
    motor_msg = xycar_motor()
    motor_msg.angle = Angle
    motor_msg.speed = Speed
    motor.publish(motor_msg)

def start():
    global motor # To assign the Publisher
    global image # To access the image from callback
    # Global state variables for lane keeping, if not passed around
    global frame_count, prev_angle, white_lost_count 

    rospy.init_node('sam_lane_keeping_xycar_driver') # Changed node name
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

    # Initialize state variables for lane keeping
    frame_count = 0
    prev_angle = 0.0
    white_lost_count = 0

    print ("---------- SAM Lane Keeping Xycar Driving Start ----------") # Updated print message
    
    # rospy.sleep(1) # Optional: give a second for ROS connections to establish

    while not rospy.is_shutdown():
        if image.size == 0:
            # time.sleep(0.02) # Optional: short sleep if no image
            continue

        # Image processing for lane keeping (ROI, HSV, Masks)
        roi = image[int(HEIGHT * 0.4):, :]
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_yellow = np.array([20, 50, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        avoidance_angle = 0.0
        avoidance_active = False
        
        # TODO: Using MOCK cone detection data from detect_cones_sam.
        cones = detect_cones_sam(image) 

        # Draw detected cones on the image
        if cones: 
            for cone in cones:
                cv2.rectangle(image, (cone['bbox'][0], cone['bbox'][1]), (cone['bbox'][2], cone['bbox'][3]), (0, 255, 0), 2)
                cv2.circle(image, (cone['center_of_mass'][0], cone['center_of_mass'][1]), 5, (0, 0, 255), -1)

            # TODO: Using MOCK data for vehicle pose and obstacle world position.
            # In a real system, pos_obstacle_world would be determined for each specific cone
            # based on its 2D detection and depth information, transformed to world coordinates.
            # The current mock function provides a single generic obstacle position.
            pos_obstacle_world, pos_vehicle, vel_vehicle = get_cone_distance_and_vehicle_pose()

            phi, active = calculate_collision_cone_avoidance(
                pos_obstacle_world,
                pos_vehicle,
                vel_vehicle,
                VEHICLE_RADIUS,
                CONE_RADIUS,
                SAFETY_MARGIN
            )

            if active:
                avoidance_angle = phi
                avoidance_active = True
        
        current_angle = 0.0
        current_speed = 60 
        current_direction = "INIT" 

        if avoidance_active:
            current_angle = np.degrees(avoidance_angle) 
            current_speed = 40  
            current_direction = "AVOIDING CONE"
        else:
            if frame_count <= 50:
                current_angle = 0
                current_speed = 60 
                current_direction = "INIT STRAIGHT"
            else:
                total_white = cv2.countNonZero(white_mask)
                if total_white > 300: 
                    white_lost_count = 0
                    left_mask  = white_mask[:, :roi_w//3]
                    right_mask = white_mask[:, 2*roi_w//3:]
                    
                    left_ratio = 0.0
                    if left_mask.size > 0: left_ratio  = cv2.countNonZero(left_mask) / float(left_mask.size)
                    right_ratio = 0.0
                    if right_mask.size > 0: right_ratio = cv2.countNonZero(right_mask) / float(right_mask.size)

                    error = (left_ratio - right_ratio) * 100
                    current_angle = np.clip(error * 0.9, -35, 35)
                    current_direction = "WHITE TRACK"
                    current_speed = 50 

                    white_nonzero = cv2.findNonZero(white_mask)
                    if white_nonzero is not None:
                        x_coords = white_nonzero[:, 0, 0]
                        min_x = np.min(x_coords)
                        max_x = np.max(x_coords)
                        track_center = (min_x + max_x) // 2
                        center_offset = (roi_w // 2) - track_center 
                        current_angle += center_offset * 0.1 
                        current_angle = np.clip(current_angle, -35, 35)

                        if max_x - min_x < roi_w * 0.35: 
                            current_direction += " | OUTSIDE WARNING"
                            current_speed = 35 
                else: 
                    white_lost_count += 1
                    M = cv2.moments(yellow_mask)
                    if M['m00'] > 0: 
                        cx = int(M['m10'] / M['m00'])
                        error = cx - (roi_w // 2) 
                        current_angle = np.clip(error * 0.005, -25, 25) 
                        current_direction = "YELLOW FALLBACK"
                    else: 
                        current_angle = prev_angle * 0.5 
                        current_direction = "NO LINE"
                    current_speed = 45 

        final_angle = current_angle
        final_speed = current_speed

        max_delta = 3  
        if abs(final_angle - prev_angle) > max_delta:
            final_angle = prev_angle + np.sign(final_angle - prev_angle) * max_delta
        
        prev_angle = final_angle 

        if not avoidance_active:
            abs_final_angle = abs(final_angle)
            if abs_final_angle < 5: final_speed = max(current_speed, 90)
            elif abs_final_angle < 10: final_speed = max(current_speed, 80)
            elif abs_final_angle < 20: final_speed = max(current_speed, 70)
            else: final_speed = max(current_speed, 60)

            if "NO LINE" in current_direction: final_speed = 30
        
        if avoidance_active: final_speed = current_speed 

        frame_count += 1
        print(f"[INFO] Mode: {current_direction}, Angle: {final_angle:.2f}, Speed: {final_speed}")

        display_info_text = ""
        text_color = (255, 255, 0) 
        if avoidance_active:
            display_info_text = f"MODE: AVOIDING CONE | Angle: {final_angle:.2f}"
            text_color = (0, 0, 255) 
        else:
            display_info_text = f"MODE: {current_direction} | Angle: {final_angle:.2f}"
            if "NO LINE" in current_direction: text_color = (0, 165, 255) 
            elif "OUTSIDE WARNING" in current_direction: text_color = (0, 255, 255) 

        cv2.putText(image, display_info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.imshow("image", image) 
        # cv2.imshow("ROI", roi) 
        # cv2.imshow("White Mask", white_mask) 
        # cv2.imshow("Yellow Mask", yellow_mask) 
        
        drive(final_angle, final_speed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # time.sleep(0.02) # Loop rate control if needed

    rospy.spin()

if __name__ == '__main__':
    start()
