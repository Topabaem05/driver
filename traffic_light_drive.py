#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, numpy as np, os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import XycarMotor

# OpenCV 브릿지 설정
bridge = CvBridge()
cv_image = np.empty(shape=[0])

# 모터 관련 변수 초기화
motor = None
motor_msg = XycarMotor()

# 주행 제어 함수
def drive(angle, speed):
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)

# 이미지 콜백 함수
def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

# 신호등 감지 함수
def detect_traffic_light(image):
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        print("경고: 유효하지 않은 이미지 입력")
        return None, None, None, None, None, None, None
        
    height, width = image.shape[:2]
    
    # 상단 부분만 ROI로 설정
    roi_height = int(height * 0.4)
    roi = image[0:roi_height, 0:width]
    
    # ROI 표시용 이미지
    roi_display = image.copy()
    # ROI 영역 표시 (녹색 사각형)
    cv2.rectangle(roi_display, (0, 0), (width, roi_height), (0, 255, 0), 2)
    
    # 이미지 전처리
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 (두 범위로 나뉨)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 노란색 범위
    lower_yellow = np.array([15, 70, 70])
    upper_yellow = np.array([35, 255, 255])
    
    # 초록색 범위
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # 각 색상 마스크 생성
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 노란색 마스크 생성 및 형태학적 연산 적용
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_DILATE, kernel)
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 각 색상의 픽셀 수 계산
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    
    # 픽셀 임계값
    threshold_red = 300
    threshold_yellow = 200
    threshold_green = 200
    
    # 결과 이미지 생성
    result_image = image.copy()
    
    # 감지된 색상 정보 표시
    speed_value = 0  # 기본값은 정지
    if red_pixels > threshold_red:
        cv2.putText(result_image, "RED - STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("빨간색 신호 감지 - 정지")
    elif yellow_pixels > threshold_yellow:
        cv2.putText(result_image, "YELLOW - STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print("노란색 신호 감지 - 정지")
    elif green_pixels > threshold_green:
        cv2.putText(result_image, "GREEN - GO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("초록색 신호 감지 - 출발")
        speed_value = 100  # 출발
    else:
        cv2.putText(result_image, "NO SIGNAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        print("신호 감지되지 않음")
    
    # 각 색상의 마스크 시각화
    # 각 색상의 마스크 합쳐서 시각화 (전체 이미지 크기로)
    mask_viz = np.zeros_like(image)
    # ROI 영역에만 마스크 적용
    mask_viz_roi = mask_viz[0:roi_height, 0:width]
    mask_roi_red = np.zeros_like(roi)
    mask_roi_red[mask_red > 0] = [0, 0, 255]  # 빨간색
    mask_roi_yellow = np.zeros_like(roi)
    mask_roi_yellow[mask_yellow > 0] = [0, 255, 255]  # 노란색
    mask_roi_green = np.zeros_like(roi)
    mask_roi_green[mask_green > 0] = [0, 255, 0]  # 초록색
    
    # 마스크 합치기
    mask_roi_combined = cv2.bitwise_or(cv2.bitwise_or(mask_roi_red, mask_roi_yellow), mask_roi_green)
    mask_viz_roi[:] = mask_roi_combined
    
    # 디버깅을 위한 개별 색상 마스크
    mask_red_viz = np.zeros_like(image)
    mask_red_viz[0:roi_height, 0:width][mask_red > 0] = [0, 0, 255]
    
    mask_yellow_viz = np.zeros_like(image)
    mask_yellow_viz[0:roi_height, 0:width][mask_yellow > 0] = [0, 255, 255]
    
    mask_green_viz = np.zeros_like(image)
    mask_green_viz[0:roi_height, 0:width][mask_green > 0] = [0, 255, 0]
    
    # 노란색 원본 영역 시각화
    yellow_original = np.zeros_like(image)
    np.copyto(yellow_original[0:roi_height, 0:width], roi, where=mask_yellow[:,:,np.newaxis]>0)
    
    return speed_value, result_image, mask_viz, mask_red_viz, mask_yellow_viz, mask_green_viz, roi_display, yellow_original

# 메인 함수
def main():
    global motor
    
    # ROS 노드 초기화
    rospy.init_node('traffic_light_driver')
    
    # 모터 퍼블리셔 설정
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
    
    # 카메라 구독
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
    
    print("트래픽 라이트 주행 시스템 시작")
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("카메라 준비 완료")
    
    # OpenCV 창 설정
    try:
        cv2.namedWindow("원본", cv2.WINDOW_NORMAL)
        cv2.namedWindow("ROI 영역", cv2.WINDOW_NORMAL)
        cv2.namedWindow("신호등 감지", cv2.WINDOW_NORMAL)
        cv2.namedWindow("색상 마스크", cv2.WINDOW_NORMAL)
        cv2.namedWindow("노란색 마스크", cv2.WINDOW_NORMAL)
        cv2.namedWindow("노란색 원본", cv2.WINDOW_NORMAL)
        print("OpenCV 창 생성 성공")
    except Exception as e:
        print(f"OpenCV 창 생성 오류: {e}")
    
    rate = rospy.Rate(10)  # 10Hz
    
    # 기본 주행 각도 (직진)
    drive_angle = 0.0
    
    while not rospy.is_shutdown():
        if cv_image.size != 0:
            try:
                # 신호등 감지
                results = detect_traffic_light(cv_image)
                if results[0] is None:
                    print("유효하지 않은 이미지 처리 결과")
                    continue
                    
                speed_value, result_image, mask_viz, mask_red_viz, mask_yellow_viz, mask_green_viz, roi_display, yellow_original = results
                
                # 신호등 색상에 따라 주행
                drive(drive_angle, speed_value)
                
                # 디버깅을 위한 이미지 표시
                try:
                    cv2.imshow("원본", cv_image)
                    cv2.imshow("ROI 영역", roi_display)
                    cv2.imshow("신호등 감지", result_image)
                    cv2.imshow("색상 마스크", mask_viz)
                    cv2.imshow("노란색 마스크", mask_yellow_viz)
                    cv2.imshow("노란색 원본", yellow_original)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"이미지 표시 오류: {e}")
                    
            except Exception as e:
                print(f"오류 발생: {e}")
        
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        # 종료 시 모터 정지
        if motor is not None:
            drive(0, 0)
        # OpenCV 창 닫기
        cv2.destroyAllWindows() 