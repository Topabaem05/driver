#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from collections import deque
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge

bridge = CvBridge()
image = None
motor_pub = None
frame_count = 0
prev_angle = 0.0
white_lost_count = 0
EMA_ALPHA = 0.3 # Exponential Moving Average alpha for angle smoothing
smoothed_angle = 0.0 #  EMA로 스무딩된 각도

# NMPC-inspired proactive control additions
RAW_ANGLE_HISTORY_SIZE = 5
raw_angle_history = deque(maxlen=RAW_ANGLE_HISTORY_SIZE)
SUSTAINED_TURN_THRESHOLD_AVG_ANGLE = 8.0 # degrees, average raw angle to detect a curve
MAX_SPEED_REDUCTION_DUE_TO_CURVE = 0.35 # Reduce speed by up to 35% for sharpest predicted curves
CURVE_NORMALIZATION_ANGLE = 25.0 # A sustained raw angle of 25deg is considered a sharp curve for max reduction

# Lateral acceleration limit based control additions
AY_MAX = 2.5  # m/s^2, 최대 허용 횡가속도 (Xycar에 맞게 튜닝 필요)
WHEELBASE = 0.26  # meters, Xycar 축거 (정확한 값으로 수정 필요)
SPEED_TO_MPS_FACTOR = 0.05 # speed 값 (0-100)을 m/s로 변환하는 계수 (매우 중요, 실험적 튜닝 필요)
                          # 예: speed 50일 때 50*0.05 = 2.5 m/s

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
    global smoothed_angle
    global raw_angle_history # deque 사용 선언

    rospy.init_node('xycar_robust')
    motor_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
    rospy.sleep(1.0)

    print("▶▶▶ Reactive Fast Driving Mode Start (with Proactive Speed Control)")

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
        # 기본 속도 설정 (나중에 코너 예측에 따라 조정될 수 있음)
        current_base_speed = 78 # 일반 주행 시 기본 속도 (60 * 1.3 = 78)
        speed = current_base_speed 
        angle = 0
        is_exiting_corner = False 
        is_predicting_curve = False # is_predicting_curve 초기화 추가

        calculated_angle_this_frame = 0 

        if frame_count <= 50: # INIT_STRAIGHT_FRAMES 와 유사
            angle = 0
            speed = 78 # 초기 직진 속도 (60 * 1.3 = 78)
            direction = "INIT STRAIGHT"
            smoothed_angle = 0.0
            raw_angle_history.clear() # 초기화 시 이력도 비움
        else:
            total_white = cv2.countNonZero(white_mask)

            if total_white > 300:
                white_lost_count = 0
                left_mask  = white_mask[:, :width//3]
                right_mask = white_mask[:, 2*width//3:]
                left_ratio  = cv2.countNonZero(left_mask) / left_mask.size
                right_ratio = cv2.countNonZero(right_mask) / right_mask.size
                error = (left_ratio - right_ratio) * 100
                calculated_angle_this_frame = np.clip(error * 0.9, -35, 35)
                direction = "WHITE TRACK"
                current_base_speed = 91 # 흰색 차선 추종 시 기본 속도 (70 * 1.3 = 91)

                white_nonzero = cv2.findNonZero(white_mask)
                if white_nonzero is not None:
                    x_coords = white_nonzero[:, 0, 0]
                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    track_center = (min_x + max_x) // 2
                    center_offset = (width // 2) - track_center
                    calculated_angle_this_frame += center_offset * 0.1
                    if max_x - min_x < width * 0.35:
                        direction += " | OUTSIDE WARNING"
                        current_base_speed = 59 # 바깥쪽 경고 시 감속 (45 * 1.3 = 58.5 ≈ 59)
            else:
                white_lost_count += 1
                M = cv2.moments(yellow_mask)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    error = cx - (width // 2)
                    # Kp 값 0.005 -> 0.25 로 수정 제안 (이전 논의 기반)
                    calculated_angle_this_frame = np.clip(error * 0.25, -25, 25) 
                    direction = "YELLOW FALLBACK"
                    current_base_speed = 72 # 노란색 차선 폴백 시 기본 속도 (55 * 1.3 = 71.5 ≈ 72)
                else:
                    calculated_angle_this_frame = prev_angle * 0.5
                    direction = "NO LINE"
                    current_base_speed = 39 # 차선 없을 시 최저 속도 (30 * 1.3 = 39)
            
            raw_angle_history.append(calculated_angle_this_frame)

            # NMPC-inspired: Proactive speed control based on predicted curve
            proactive_speed = current_base_speed
            is_predicting_curve = False # 코너 예측 플래그
            avg_raw_angle_abs = 0 # 코너 탈출 감지에도 사용하기 위해 스코프 확장

            if len(raw_angle_history) == RAW_ANGLE_HISTORY_SIZE:
                avg_raw_angle = np.mean(list(raw_angle_history))
                avg_raw_angle_abs = abs(avg_raw_angle) # 여기서 계산

                if avg_raw_angle_abs > SUSTAINED_TURN_THRESHOLD_AVG_ANGLE:
                    is_predicting_curve = True
                    curve_severity_ratio = min(avg_raw_angle_abs / CURVE_NORMALIZATION_ANGLE, 1.0)
                    speed_reduction_factor = curve_severity_ratio * MAX_SPEED_REDUCTION_DUE_TO_CURVE
                    proactive_speed = current_base_speed * (1.0 - speed_reduction_factor)
                    direction += f" | CURVE PRED ({(1.0 - speed_reduction_factor)*100:.0f}% SPD)"
            
            speed_before_lat_limit = proactive_speed

            # Lateral G-force limit based speed adjustment
            current_speed_mps = speed_before_lat_limit * SPEED_TO_MPS_FACTOR
            target_steer_angle_rad = np.deg2rad(calculated_angle_this_frame)
            K_des = 0 # K_des 초기화
            if WHEELBASE > 1e-3:
                K_des = target_steer_angle_rad / WHEELBASE
            
            # 코너 탈출 감지 로직 추가
            if is_predicting_curve and \
               abs(smoothed_angle) < SUSTAINED_TURN_THRESHOLD_AVG_ANGLE * 0.8 and \
               abs(calculated_angle_this_frame) < avg_raw_angle_abs * 0.7:
                is_exiting_corner = True
                direction += " | EXITING CURVE"

            speed = speed_before_lat_limit # 기본값으로 이전 단계 속도 할당
            if abs(K_des) > 1e-6:
                max_speed_for_this_curvature_mps = np.sqrt(AY_MAX / (abs(K_des) + 1e-9))
                if current_speed_mps > max_speed_for_this_curvature_mps:
                    speed = max(0, max_speed_for_this_curvature_mps / (SPEED_TO_MPS_FACTOR + 1e-9))
                    direction += f" | LAT G LIMIT (to {speed:.0f})"
            # (else: K_des가 거의 0이면 speed_before_lat_limit가 speed로 유지됨)
            
            # EMA 필터를 사용하여 angle 스무딩
            if frame_count == 51:
                smoothed_angle = calculated_angle_this_frame
            else:
                smoothed_angle = EMA_ALPHA * calculated_angle_this_frame + (1 - EMA_ALPHA) * smoothed_angle
            
            angle = smoothed_angle

        # 조향 변화 제한 (최대 부드럽게)
        max_delta = 3
        if abs(angle - prev_angle) > max_delta:
            angle = prev_angle + np.sign(angle - prev_angle) * max_delta
        prev_angle = angle

        # 최종 속도 결정
        abs_final_angle = abs(angle)
        
        if is_exiting_corner:
            if abs_final_angle < 5: 
                speed = max(speed, current_base_speed * 1.1, 90) 
            elif abs_final_angle < 10: 
                speed = max(speed, current_base_speed * 1.05)
            else: 
                speed = max(speed, current_base_speed)
            direction += " | ACCEL POST-CURVE"
        elif not is_predicting_curve:
            if abs_final_angle < 5:   
                speed = max(speed, 91)
            elif abs_final_angle < 10: 
                speed = max(speed, 104)
            elif abs_final_angle < 20: 
                speed = max(speed, 91)
            else:                      
                speed = max(speed, 78)
        
        if "NO LINE" in direction:
            speed = 39
        
        speed = np.clip(speed, 30, 110)

        print(f"[INFO] Mode: {direction}, Angle: {angle:.2f}, Speed: {int(speed)}")
        cv2.imshow("original", image)
        cv2.imshow("white_mask", white_mask)
        cv2.imshow("yellow_mask", yellow_mask)
        cv2.waitKey(1)

        drive(angle, speed)
        time.sleep(0.02)

if __name__ == '__main__':
    start()
