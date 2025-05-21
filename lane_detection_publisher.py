#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Carla-ROS Lane Keeping System - Lane Detection Publisher
- LaneNet 기반 차선 검출 및 ROS 토픽 퍼블리시
- Carla 시뮬레이터 이미지 구독 및 처리
- 차선 중심점 계산 및 조향 제어 정보 제공
"""

import rospy
import numpy as np
import cv2
import os
import sys
import tensorflow as tf
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# 현재 디렉토리 경로 설정
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_PATH, "models"))
sys.path.append(os.path.join(ROOT_PATH, "models/LaneNet"))

# LaneNet 모델 관련 임포트
import lanenet
import parse_config_utils


class LaneNet(object):
    """LaneNet 딥러닝 모델 클래스"""
    
    def __init__(self):
        """LaneNet 모델 초기화"""
        # 설정 파일 로드
        self.cfg = parse_config_utils.lanenet_cfg
        
        # 텐서플로우 입력 텐서 정의
        self.input_tensor = tf.placeholder(
            dtype=tf.float32, 
            shape=[1, 256, 512, 3], 
            name='input_tensor'
        )
        
        # LaneNet 모델 초기화
        self.net = lanenet.LaneNet(phase='test', cfg=self.cfg)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(
            input_tensor=self.input_tensor, 
            name='LaneNet'
        )
        
        # 모델 가중치 경로 설정
        self.weights_path = os.path.join(
            ROOT_PATH, 
            "models/LaneNet/weights/tusimple_lanenet.ckpt"
        )
        
        # 텐서플로우 세션 설정
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = self.cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self.cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=sess_config)
        
        # 학습된 변수 로드
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                self.cfg.SOLVER.MOVING_AVE_DECAY
            )
            variables_to_restore = variable_averages.variables_to_restore()
            
        self.saver = tf.train.Saver(variables_to_restore)
        self.saver.restore(sess=self.sess, save_path=self.weights_path)
        
        rospy.loginfo("LaneNet Model Initialized")
    
    def preProcessing(self, image):
        """이미지 전처리 함수"""
        # 이미지 크기 조정 및 정규화
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image
    
    def predict(self, image):
        """이미지 예측 함수"""
        # 이미지 전처리
        src_image = self.preProcessing(image)
        
        # 모델 추론
        with self.sess.as_default():
            binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [src_image]}
            )
        
        # 결과 후처리
        rgb = instance_seg_image[0].astype(np.uint8)
        bw = binary_seg_image[0].astype(np.uint8)
        res = cv2.bitwise_and(rgb, rgb, mask=bw)
        
        # 차선 및 중심점 계산
        lanes_rgb, center_xy = self.postProcess(res)
        
        return lanes_rgb, center_xy
    
    def postProcess(self, image):
        """예측 결과 후처리 함수"""
        # RGB 변환
        src_img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 빨간색 마스크 제거 (배경)
        red_mask = (src_img[:,:,2] > 200).astype(np.uint8)
        src_img = cv2.bitwise_and(src_img, src_img, mask=1-red_mask)
        
        # 오른쪽 차선 (녹색)
        green_mask = (src_img[:,:,1] > 200).astype(np.uint8)
        green_area = cv2.bitwise_and(src_img, src_img, mask=green_mask)
        
        # 왼쪽 차선 (파란색)
        blue_mask = (src_img[:,:,0] > 200).astype(np.uint8)
        blue_area = cv2.bitwise_and(src_img, src_img, mask=blue_mask)
        
        # 차선 이미지 합성
        lanes_rgb = cv2.addWeighted(green_area, 1, blue_area, 1, 0)
        
        # 차선 중심점 계산
        img_center_point, center_xy = self.window_search(green_mask, blue_mask)
        lanes_rgb = cv2.addWeighted(lanes_rgb, 1, img_center_point, 1, 0)
        
        return lanes_rgb, center_xy
    
    @staticmethod
    def window_search(right_lane, left_lane):
        """슬라이딩 윈도우 기반 차선 중심점 계산 함수"""
        center_coordinates = []
        out = np.zeros(right_lane.shape, np.uint8)
        out = cv2.merge((out, out, out))
        
        # 이미지 중심점 계산
        mid_point = np.int(right_lane.shape[1]/2)
        
        # 윈도우 설정
        nwindows = 9
        h = right_lane.shape[0]
        vp = int(h/2)  # 소실점(vanishing point)
        window_height = np.int(vp/nwindows)
        
        # 차선 마스크 처리
        r_lane = right_lane[vp:,:].copy()
        r_lane = cv2.erode(r_lane, np.ones((3,3)))
        l_lane = left_lane[vp:,:]
        l_lane = cv2.erode(l_lane, np.ones((3,3)))
        
        # 각 윈도우별 중심점 계산
        for window in range(nwindows):
            win_y_low = vp - (window+1)*window_height
            win_y_high = vp - window*window_height
            win_y_center = win_y_low + int((win_y_high-win_y_low)/2)
            
            # 각 윈도우 영역 내 차선 검출
            r_row = r_lane[win_y_low:win_y_high,:]
            l_row = l_lane[win_y_low:win_y_high,:]
            
            # 히스토그램 기반 차선 위치 검출
            histogram = np.sum(r_row, axis=0)
            r_point = np.argmax(histogram)
            
            histogram = np.sum(l_row, axis=0)
            l_point = np.argmax(histogram)
            
            # 좌우 차선이 모두 검출된 경우
            if (l_point != 0) and (r_point != 0):
                rd = r_point - mid_point
                ld = mid_point - l_point
                
                # 좌우 차선 간격이 적절한 경우
                if abs(rd-ld) < 100:
                    # 중심점 계산 및 표시
                    center = l_point + int((r_point-l_point)/2)
                    out = cv2.circle(out, (center, vp+win_y_center), 2, (0,0,255), -1)
                    center_coordinates.append((center, vp+win_y_center))
        
        return out, center_coordinates


class LaneDetection(object):
    """차선 검출 및 ROS 노드 클래스"""
    
    def __init__(self):
        """ROS 노드 초기화"""
        # ROS 노드 초기화
        rospy.init_node('lane_detection', anonymous=True)
        
        # LaneNet 모델 초기화
        self.model = LaneNet()
        
        # OpenCV 브릿지 초기화
        self.bridge = CvBridge()
        
        # 디버그 모드 설정
        self.debug_mode = rospy.get_param('~debug', True)
        
        # 차선 중심점 관련 변수
        self.lane_center_x = None
        self.image_center_x = None
        self.steering_angle = 0.0
        
        # PID 제어 관련 변수
        self.kp = 0.01  # 비례 게인
        self.ki = 0.0001  # 적분 게인
        self.kd = 0.01  # 미분 게인
        self.prev_error = 0
        self.integral = 0
        
        # ROS 토픽 구독 및 발행
        rospy.Subscriber(
            "/carla/ego_vehicle/rgb_front/image", 
            Image, 
            self.image_callback
        )
        
        self.image_pub = rospy.Publisher(
            "/lka/detected_image",
            Image,
            queue_size=1
        )
        
        self.steering_pub = rospy.Publisher(
            "/lka/steering_angle",
            Float32,
            queue_size=1
        )
        
        rospy.loginfo("Lane Detection Node Initialized")
    
    def image_callback(self, raw_image):
        """이미지 콜백 함수"""
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            byte_image = raw_image.data
            np_image = np.frombuffer(byte_image, dtype=np.uint8)
            bgra_image = np_image.reshape((raw_image.height, raw_image.width, 4))
            rgb_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)
            
            # 이미지 중심점 저장
            self.image_center_x = rgb_image.shape[1] // 2
            
            # LaneNet 모델로 차선 검출
            prediction, lane_center = self.model.predict(rgb_image)
            
            # 차선 중심점 계산 및 조향각 계산
            self.calculate_steering(lane_center)
            
            # 결과 이미지에 정보 표시
            if self.debug_mode and prediction is not None:
                # 조향각 정보 표시
                cv2.putText(
                    prediction,
                    f"Steering: {self.steering_angle:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # 차선 중심점과 이미지 중심점 표시
                if self.lane_center_x is not None:
                    cv2.line(
                        prediction,
                        (self.lane_center_x, 0),
                        (self.lane_center_x, prediction.shape[0]),
                        (0, 255, 255),
                        2
                    )
                
                cv2.line(
                    prediction,
                    (self.image_center_x, 0),
                    (self.image_center_x, prediction.shape[0]),
                    (255, 0, 0),
                    2
                )
            
            # 결과 이미지 발행
            publish_image = Image()
            publish_image.header = raw_image.header
            publish_image.is_bigendian = raw_image.is_bigendian
            publish_image.encoding = raw_image.encoding
            publish_image.height = prediction.shape[0]
            publish_image.width = prediction.shape[1]
            
            # 이미지 포맷 변환 및 발행
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGRA).astype(np.uint8)
            byte_data = prediction.tobytes()
            publish_image.data = byte_data
            self.image_pub.publish(publish_image)
            
            # 조향각 발행
            self.steering_pub.publish(Float32(self.steering_angle))
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
    
    def calculate_steering(self, lane_center):
        """차선 중심점 기반 조향각 계산 함수"""
        if not lane_center or len(lane_center) == 0:
            # 차선이 검출되지 않은 경우
            rospy.logwarn("No lane centers detected")
            return
        
        # 가장 아래쪽(가까운) 차선 중심점 사용
        closest_center = sorted(lane_center, key=lambda x: x[1], reverse=True)
        if closest_center:
            self.lane_center_x = closest_center[0][0]
            
            # 차선 중심과 이미지 중심 간 오차 계산
            error = self.lane_center_x - self.image_center_x
            
            # PID 제어
            self.integral += error
            derivative = error - self.prev_error
            
            # 조향각 계산 (PID)
            self.steering_angle = (
                self.kp * error + 
                self.ki * self.integral + 
                self.kd * derivative
            )
            
            # 조향각 제한 (-1.0 ~ 1.0)
            self.steering_angle = np.clip(self.steering_angle, -1.0, 1.0)
            
            # 이전 오차 저장
            self.prev_error = error


def main():
    """메인 함수"""
    try:
        lane_detection = LaneDetection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")


if __name__ == "__main__":
    main()
