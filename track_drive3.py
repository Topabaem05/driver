#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge

bridge = CvBridge()
image = None
motor_pub = None
frame_count = 0
prev_angle = 0.0
white_lost_count = 0

def img_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def drive(angle, speed):
    msg = XycarMotor()
    msg.angle = float(angle)
    msg.speed = float(speed)
    motor_pub.publish(msg)

def start():
    global motor_pub, frame_count, prev_angle, white_lost_count

    rospy.init_node('xycar_robust')
    motor_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    rospy.sleep(1.0)

    print("▶▶▶ Reactive Fast Driving Mode Start")

    while not rospy.is_shutdown():
        if image is None:
            continue

        height, width = image.shape[:2]
        roi = image[int(height * 0.4):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        frame_count += 1
        direction = ""
        speed = 60
        angle = 0

        if frame_count <= 50:
            angle = 0
            speed = 60
            direction = "INIT STRAIGHT"
        else:
            total_white = cv2.countNonZero(white_mask)

            if total_white > 300:
                white_lost_count = 0

                left_mask  = white_mask[:, :width//3]
                right_mask = white_mask[:, 2*width//3:]
                left_ratio  = cv2.countNonZero(left_mask) / left_mask.size
                right_ratio = cv2.countNonZero(right_mask) / right_mask.size

                error = (left_ratio - right_ratio) * 100
                angle = np.clip(error * 0.9, -35, 35)

                direction = "WHITE TRACK"

                white_nonzero = cv2.findNonZero(white_mask)
                if white_nonzero is not None:
                    x_coords = white_nonzero[:, 0, 0]
                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    track_center = (min_x + max_x) // 2
                    center_offset = (width // 2) - track_center
                    angle += center_offset * 0.1

                    if max_x - min_x < width * 0.35:
                        direction += " | OUTSIDE WARNING"
                        speed = 35
            else:
                white_lost_count += 1
                M = cv2.moments(yellow_mask)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    error = cx - (width // 2)
                    angle = np.clip(error * 0.005, -25, 25)
                    direction = "YELLOW FALLBACK"
                else:
                    angle = 0
                    direction = "NO LINE"
                speed = 45

        # 조향 변화 제한 (최대 부드럽게)
        max_delta = 3  # 조향 전환 부드럽게
        if abs(angle - prev_angle) > max_delta:
            angle = prev_angle + np.sign(angle - prev_angle) * max_delta
        prev_angle = angle

        abs_angle = abs(angle)
        if abs_angle < 5:
            speed = max(speed, 90)
        elif abs_angle < 10:
            speed = max(speed, 80)
        elif abs_angle < 20:
            speed = max(speed, 70)
        else:
            speed = max(speed, 60)

        if "NO LINE" in direction:
            speed = 30

        print(f"[INFO] Mode: {direction}, Angle: {angle:.2f}, Speed: {speed}")
        cv2.imshow("original", image)
        cv2.imshow("white_mask", white_mask)
        cv2.imshow("yellow_mask", yellow_mask)
        cv2.waitKey(1)

        drive(angle, speed)
        time.sleep(0.02)

if __name__ == '__main__':
    start()




