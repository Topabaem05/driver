#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge

class LaneDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.prev_angle = 0.0
        self.frame_count = 0

        # Bird Eye View 변환 설정
        self.src_points = None
        self.dst_points = None
        self.M = None
        self.Minv = None

        # 슬라이딩 윈도우 설정
        self.nwindows = 9
        self.window_width = 100
        self.min_pixels = 50

        # 차선 검출 상태 추적
        self.left_fit = None
        self.right_fit = None
        self.left_detected = False
        self.right_detected = False
        self.white_lost_count = 0

        # Kalman 필터 설정
        self.use_kalman = True
        self.kalman_left = cv2.KalmanFilter(6, 3)
        self.kalman_right = cv2.KalmanFilter(6, 3)
        self.setup_kalman_filter()

        # ROS 설정
        self.motor_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
        rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback)

        # 디버깅 설정
        self.debug = True

    def setup_kalman_filter(self):
        self.kalman_left.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kalman_right.transitionMatrix = self.kalman_left.transitionMatrix.copy()
        self.kalman_left.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        self.kalman_right.measurementMatrix = self.kalman_left.measurementMatrix.copy()
        self.kalman_left.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kalman_right.processNoiseCov = self.kalman_left.processNoiseCov.copy()
        self.kalman_left.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        self.kalman_right.measurementNoiseCov = self.kalman_left.measurementNoiseCov.copy()
        self.kalman_left.errorCovPost = np.eye(6, dtype=np.float32)
        self.kalman_right.errorCovPost = self.kalman_left.errorCovPost.copy()

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def drive(self, angle, speed):
        msg = XycarMotor()
        msg.angle = float(angle)
        msg.speed = float(speed)
        self.motor_pub.publish(msg)

    def calculate_bird_eye_view_matrix(self, img_shape):
        height, width = img_shape[:2]
        # 소스 포인트: 이미지 하단 영역을 더 넓게, 상단은 좁게 (원본 이미지 기준)
        # 참고 코드의 ROI: image[int(height*0.6):, :]
        # 이를 BEV 소스 포인트에 반영
        self.src_points = np.float32([
            [width * 0.1, height * 0.95],       # 왼쪽 아래
            [width * 0.45, height * 0.6],      # 왼쪽 위
            [width * 0.55, height * 0.6],      # 오른쪽 위
            [width * 0.9, height * 0.95]        # 오른쪽 아래
        ])
        # 목적지 포인트: BEV 이미지의 전체 너비와 높이를 사용하도록 설정
        self.dst_points = np.float32([
            [0, height],                    # 왼쪽 아래
            [0, 0],                         # 왼쪽 위
            [width, 0],                     # 오른쪽 위
            [width, height]                 # 오른쪽 아래
        ])
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def apply_bird_eye_view(self, img):
        if self.M is None or self.Minv is None:
             # 이미지 모양 전달
            self.calculate_bird_eye_view_matrix(img.shape) 
        # 전체 이미지에 대해 BEV 변환 수행
        height, width = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (width, height), flags=cv2.INTER_LINEAR)

    def create_binary_image(self, bev_img):
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180]) # 참고 코드 값
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([20, 100, 100]) # 참고 코드 값
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        # Morphological operations to remove noise - optional, can be added if needed
        # kernel = np.ones((3,3),np.uint8)
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        return combined_mask, white_mask, yellow_mask

    def find_lane_pixels_sliding_window(self, binary_warped):
        height, width = binary_warped.shape
        histogram = np.sum(binary_warped[height//2:, :], axis=0)
        out_img_debug = None
        if self.debug:
            out_img_debug = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        midpoint = width // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        if not histogram[:midpoint].any(): leftx_base = width // 4
        if not histogram[midpoint:].any(): rightx_base = width * 3 // 4
            
        window_height = height // self.nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(self.nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = max(0, leftx_current - self.window_width // 2)
            win_xleft_high = min(width, leftx_current + self.window_width // 2)
            win_xright_low = max(0, rightx_current - self.window_width // 2)
            win_xright_high = min(width, rightx_current + self.window_width // 2)

            if self.debug and out_img_debug is not None:
                cv2.rectangle(out_img_debug, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
                cv2.rectangle(out_img_debug, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.min_pixels: leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_pixels: rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])
            
        leftx, lefty = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds]) if len(left_lane_inds) > 0 else (np.array([]), np.array([]))
        rightx, righty = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds]) if len(right_lane_inds) > 0 else (np.array([]), np.array([]))

        self.left_detected = len(leftx) > self.min_pixels
        self.right_detected = len(rightx) > self.min_pixels
        return leftx, lefty, rightx, righty, out_img_debug

    def fit_polynomial(self, leftx, lefty, rightx, righty):
        if self.left_detected and len(lefty) > 0 and len(leftx) > 0:
            left_fit_new = np.polyfit(lefty, leftx, 2)
            if self.use_kalman and self.left_fit is not None:
                measurement = np.array([[left_fit_new[0]], [left_fit_new[1]], [left_fit_new[2]]], np.float32)
                self.kalman_left.correct(measurement)
                self.left_fit = self.kalman_left.predict()[:3].reshape(-1)
            else:
                self.left_fit = left_fit_new
        elif not self.left_detected: self.left_fit = None
        
        if self.right_detected and len(righty) > 0 and len(rightx) > 0:
            right_fit_new = np.polyfit(righty, rightx, 2)
            if self.use_kalman and self.right_fit is not None:
                measurement = np.array([[right_fit_new[0]], [right_fit_new[1]], [right_fit_new[2]]], np.float32)
                self.kalman_right.correct(measurement)
                self.right_fit = self.kalman_right.predict()[:3].reshape(-1)
            else:
                self.right_fit = right_fit_new
        elif not self.right_detected: self.right_fit = None
        return self.left_fit, self.right_fit

    def calculate_curvature_and_position(self, left_fit, right_fit, img_shape):
        ym_per_pix, xm_per_pix = 30/720, 3.7/700
        height, width = img_shape[:2]
        ploty = np.linspace(0, height - 1, height)
        y_eval = height - 1 
        left_curverad, right_curverad = float('inf'), float('inf')
        center_offset_meters = 0

        if left_fit is not None:
            leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            if len(ploty)>1 and len(leftx)>1: # Need at least 2 points for polyfit
                left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
                left_curverad = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0]) if abs(2*left_fit_cr[0]) > 1e-6 else float('inf')

        if right_fit is not None:
            rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            if len(ploty)>1 and len(rightx)>1:
                right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
                right_curverad = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0]) if abs(2*right_fit_cr[0]) > 1e-6 else float('inf')

        curvature = float('inf')
        car_center_px = width // 2

        if left_fit is not None and right_fit is not None:
            if left_curverad != float('inf') and right_curverad != float('inf'): curvature = (left_curverad + right_curverad) / 2
            else: curvature = max(left_curverad, right_curverad) # Use one if the other is inf
            left_x_base = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            right_x_base = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            lane_center_px = (left_x_base + right_x_base) / 2
            center_offset_meters = (lane_center_px - car_center_px) * xm_per_pix
        elif left_fit is not None:
            curvature = left_curverad
            left_x_base = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            # Estimate lane center assuming a fixed lane width to the right of left lane
            estimated_lane_center_px = left_x_base + ( (3.7 / xm_per_pix) / 2 ) # 3.7m lane width
            center_offset_meters = (estimated_lane_center_px - car_center_px) * xm_per_pix
        elif right_fit is not None:
            curvature = right_curverad
            right_x_base = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
             # Estimate lane center assuming a fixed lane width to the left of right lane
            estimated_lane_center_px = right_x_base - ( (3.7 / xm_per_pix) / 2 )
            center_offset_meters = (estimated_lane_center_px - car_center_px) * xm_per_pix
        return curvature, center_offset_meters

    def draw_lane_area(self, original_img, warped_binary_img_shape, left_fit, right_fit):
        h, w = warped_binary_img_shape[:2]
        original_h, original_w = original_img.shape[:2]
        color_warp = np.zeros((h, w, 3), dtype=np.uint8)
        ploty = np.linspace(0, h - 1, h)
        fill_color = (0,0,0) # Default to black if no lanes

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            fill_color = (0, 255, 0) # Green for both lanes
        elif left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            # Dummy right line for visualization, assuming fixed width
            dummy_right_fitx = left_fitx + (3.7 / (3.7/700)) # 3.7m in pixels
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([dummy_right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            fill_color = (255, 200, 0) # Light blue for left only
        elif right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            dummy_left_fitx = right_fitx - (3.7 / (3.7/700))
            pts_left = np.array([np.transpose(np.vstack([dummy_left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            fill_color = (0, 200, 255) # Light orange for right only
        
        if left_fit is not None or right_fit is not None: # Check if there is anything to draw
             cv2.fillPoly(color_warp, np.int_([pts]), fill_color)

        final_result = original_img
        if self.Minv is not None:
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (original_w, original_h))
            final_result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return final_result

    def calculate_steering_angle(self, curvature, center_offset):
        angle_from_curvature = 0
        # Curvature is radius. Smaller radius -> sharper turn. Angle should be larger.
        # Inverse relation. Sign of curvature might indicate turn direction.
        # Assuming positive curvature = right turn, negative = left turn.
        # If curvature is large (straight), angle_from_curvature is small.
        if curvature != float('inf') and abs(curvature) > 1:
             # Need to determine a sensible scaling factor. E.g. 50000 / curvature
             # If curvature = 500m, angle = 10. If 250m, angle = 20.
             # The sign of curvature should dictate the sign of angle. 
             # If right turn (positive curv) -> positive angle.
             # This depends on how curvature is calculated (sign convention)
             # Let's assume our curvature calculation gives + for right, - for left
            angle_from_curvature = np.clip( (1/curvature) * 5000 , -25, 25) # Example scaling

        # Center offset: if car is to the right of center (positive offset), steer left (negative angle)
        position_correction = -center_offset * 35 # Gain for position correction
        
        angle = angle_from_curvature + position_correction
        max_delta = 15
        angle_delta = angle - self.prev_angle
        if abs(angle_delta) > max_delta:
            angle = self.prev_angle + np.sign(angle_delta) * max_delta
        self.prev_angle = angle
        return np.clip(angle, -50, 50)

    def determine_speed(self, angle):
        abs_angle = abs(angle)
        if abs_angle < 7: speed = 50 # Reduced from 60
        elif abs_angle < 15: speed = 40 # Reduced from 45
        else: speed = 30 # Reduced from 35
        if not (self.left_detected and self.right_detected): speed = max(20, speed - 10) # Reduced from 25
        return int(speed)

    def process_image(self, img_orig):
        if img_orig is None: return None, self.prev_angle, self.determine_speed(self.prev_angle)
        
        # Initialize BEV matrix based on the first frame's shape if not already done
        if self.M is None: 
            self.calculate_bird_eye_view_matrix(img_orig.shape)

        bev_img = self.apply_bird_eye_view(img_orig)
        binary_bev, white_mask_bev, yellow_mask_bev = self.create_binary_image(bev_img)

        current_angle = self.prev_angle
        direction_log = "INIT_FRAME"
        result_img_to_show = img_orig.copy()
        sliding_window_debug_img = None
        calculated_curvature = float('inf')
        calculated_offset = 0

        total_white_pixels_bev = cv2.countNonZero(white_mask_bev)

        if total_white_pixels_bev > 250: # Adjusted threshold
            self.white_lost_count = 0
            leftx, lefty, rightx, righty, sliding_window_debug_img = self.find_lane_pixels_sliding_window(binary_bev)
            
            current_left_fit, current_right_fit = self.fit_polynomial(leftx, lefty, rightx, righty)
            
            if self.left_detected or self.right_detected: # At least one lane fit is available
                calculated_curvature, calculated_offset = self.calculate_curvature_and_position(current_left_fit, current_right_fit, binary_bev.shape)
                current_angle = self.calculate_steering_angle(calculated_curvature, calculated_offset)
                direction_log = "LANE_TRACK"
                result_img_to_show = self.draw_lane_area(img_orig, binary_bev.shape, current_left_fit, current_right_fit)
            else:
                direction_log = "WHITE_PIXELS_NO_FIT"
                # Keep previous angle or use some recovery if no fit from sliding window
        else:
            self.white_lost_count += 1
            M_yellow = cv2.moments(yellow_mask_bev)
            if M_yellow['m00'] > 0:
                cx_yellow = int(M_yellow['m10'] / M_yellow['m00'])
                error_yellow = cx_yellow - (binary_bev.shape[1] // 2)
                current_angle = np.clip(error_yellow * 0.025, -30, 30) # Adjusted gain for yellow fallback
                direction_log = "YELLOW_FALLBACK"
            else:
                current_angle = -25 if self.prev_angle >= 0 else 25 # Basic recovery steer
                direction_log = "NO_LINE_RECOVERY"
        
        current_speed = self.determine_speed(current_angle)

        if self.debug:
            if sliding_window_debug_img is not None: cv2.imshow("Sliding Window (BEV)", sliding_window_debug_img)
            cv2.imshow("Binary BEV", binary_bev)
            cv2.putText(result_img_to_show, f"DIR: {direction_log}", (5,15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"ANG: {current_angle:.1f}", (5,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"SPD: {current_speed}", (5,45), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"CRV: {calculated_curvature:.0f}", (5,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"OFF: {calculated_offset:.2f}", (5,75), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"WLC: {self.white_lost_count}", (5,90), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            cv2.putText(result_img_to_show, f"L/R: {self.left_detected}/{self.right_detected}",(5,105),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        return result_img_to_show, current_angle, current_speed

    def run(self):
        rospy.sleep(1.0)
        print("▶▶▶ Lane Detection V3 Start")
        rate = rospy.Rate(15) # Target 15 Hz
        init_frames = 20 # Number of initial straight frames
        init_speed = 35  # Speed during initialization

        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue
            
            loop_start_time = time.time()
            self.frame_count += 1

            if self.frame_count <= init_frames:
                final_angle, final_speed = 0, init_speed
                display_image = self.image.copy()
                cv2.putText(display_image, "INIT", (5,15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            else:
                display_image, final_angle, final_speed = self.process_image(self.image)
            
            if display_image is not None:
                cv2.imshow("Xycar Robust Driving", display_image)
                cv2.waitKey(1)
            
            self.drive(final_angle, final_speed)
            
            # For performance monitoring
            # loop_duration = time.time() - loop_start_time
            # print(f"FPS: {1.0/loop_duration if loop_duration > 0 else 0 :.1f}")
            rate.sleep()

def main():
    rospy.init_node('xycar_lane_detector', anonymous=True)
    detector = LaneDetector()
    try:
        detector.run()
    except rospy.ROSInterruptException:
        print("Shutting down Lane Detector node.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
