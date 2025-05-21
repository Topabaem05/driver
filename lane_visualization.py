#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Carla-ROS Lane Keeping System - 시각화 노드
- 차선 검출 결과 이미지 구독 및 시각화
- 조향각 정보 시각화
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from threading import Thread
from cv_bridge import CvBridge


class LaneVisualization:
    """차선 검출 결과 시각화 클래스"""
    
    def __init__(self):
        """ROS 노드 초기화"""
        # ROS 노드 초기화
        rospy.init_node('lane_visualization', anonymous=True)
        
        # OpenCV 브릿지 초기화
        self.bridge = CvBridge()
        
        # 이미지 및 조향각 저장 변수
        self.current_image = None
        self.current_steering = 0.0
        
        # 윈도우 설정
        cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Detection", 800, 600)
        
        # ROS 토픽 구독
        rospy.Subscriber(
            "/lka/detected_image", 
            Image, 
            self.image_callback
        )
        
        rospy.Subscriber(
            "/lka/steering_angle", 
            Float32, 
            self.steering_callback
        )
        
        rospy.loginfo("Lane Visualization Node Initialized")
        
        # 디스플레이 업데이트 스레드 시작
        self.update_thread = Thread(target=self.update_display)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def image_callback(self, image_msg):
        """이미지 콜백 함수"""
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            byte_image = image_msg.data
            np_image = np.frombuffer(byte_image, dtype=np.uint8)
            bgra_image = np_image.reshape((image_msg.height, image_msg.width, 4))
            bgr_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
            
            # 현재 이미지 업데이트
            self.current_image = bgr_image
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
    
    def steering_callback(self, steering_msg):
        """조향각 콜백 함수"""
        self.current_steering = steering_msg.data
    
    def update_display(self):
        """디스플레이 업데이트 함수"""
        rate = rospy.Rate(30)  # 30Hz
        
        while not rospy.is_shutdown():
            if self.current_image is not None:
                # 이미지 복사
                display_image = self.current_image.copy()
                
                # 조향각 정보 표시
                steering_text = f"Steering: {self.current_steering:.2f}"
                cv2.putText(
                    display_image,
                    steering_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # 조향 방향 시각화
                h, w = display_image.shape[:2]
                center_x = w // 2
                center_y = h - 50
                radius = 50
                angle = -self.current_steering * np.pi / 2  # 조향각을 라디안으로 변환
                
                # 조향 방향 표시
                end_x = int(center_x + radius * np.sin(angle))
                end_y = int(center_y - radius * np.cos(angle))
                
                cv2.circle(display_image, (center_x, center_y), radius, (0, 0, 255), 2)
                cv2.line(display_image, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
                
                # 이미지 표시
                cv2.imshow("Lane Detection", display_image)
                cv2.waitKey(1)
            
            rate.sleep()


def main():
    """메인 함수"""
    try:
        visualization = LaneVisualization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
