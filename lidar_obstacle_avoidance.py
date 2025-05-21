#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

# 프로그램에서 사용할 변수, 저장공간 선언부
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
motor_msg = XycarMotor()  # 모터 토픽 메시지
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 

# 주행 관련 상수 및 변수
DEFAULT_SPEED = 10  # 기본 주행 속도 (30으로 낮춤)
OBSTACLE_DISTANCE_THRESHOLD = 0.7  # 장애물 감지 임계값(m)
FRONT_ANGLE_RANGE = 10  # 전방 간주 각도 범위 (중심에서 좌우로 ±)
SIDE_ANGLE_RANGE = 30  # 측면 간주 각도 범위

# 새로운 차선 주행 로직을 위한 전역 변수
frame_count = 0
prev_angle = 0.0
white_lost_count = 0

# 라바콘 감지 관련 상수
CONE_CLUSTER_THRESHOLD = 0.2  # 라바콘 클러스터 거리 임계값(m)
CONE_MIN_POINTS = 3  # 라바콘으로 인식하기 위한 최소 포인트 수
CONE_HEIGHT_MIN = 0.2  # 라바콘 최소 높이(m)
CONE_HEIGHT_MAX = 0.5  # 라바콘 최대 높이(m)
CONE_AVOIDANCE_GAIN = 2.5  # 라바콘 회피 계수 (회전 반응성 증가)

# 라바콘 색상 범위 (HSV)
ORANGE_LOWER = np.array([5, 100, 150])
ORANGE_UPPER = np.array([15, 255, 255])

# 차선 인식 관련 상수 (새로운 코드 기준으로 변경)
# LANE_ROI_BOTTOM = 0.85  # ROI 하단 위치 (이미지 높이의 비율) - 확장 (주석 처리)
# LANE_ROI_TOP = 0.3  # ROI 상단 위치 - 확장 (주석 처리)
# LANE_LOW_THRESHOLD = 50  # Canny 엣지 검출 임계값 (주석 처리)
# LANE_HIGH_THRESHOLD = 150 (주석 처리)
# LANE_HOUGH_THRESHOLD = 40  # 허프 변환 임계값 (주석 처리)
# LANE_MIN_LENGTH = 30  # 감지할 선의 최소 길이 (주석 처리)
# LANE_MAX_GAP = 50  # 선 간의 최대 간격 (주석 처리)

# 차선 색상 범위 (HSV) - 새로운 코드 기준
WHITE_LOWER_NEW = np.array([0, 0, 180])
WHITE_UPPER_NEW = np.array([180, 30, 255])
YELLOW_LOWER_NEW = np.array([20, 100, 100])
YELLOW_UPPER_NEW = np.array([30, 255, 255])

# 기존 차선 색상 범위 (필요시 다른 기능에서 사용될 수 있으므로 일단 유지 또는 주석처리)
WHITE_LOWER_OLD = np.array([0, 0, 180])
WHITE_UPPER_OLD = np.array([180, 40, 255])
YELLOW_LOWER_OLD = np.array([15, 60, 80])
YELLOW_UPPER_OLD = np.array([40, 255, 255])

# 주행 모드 관련 변수
MODE_LANE_FOLLOWING = 0  # 차선 추종 모드
MODE_OBSTACLE_AVOIDANCE = 1  # 장애물 회피 모드
MODE_CONE_AVOIDANCE = 2  # 라바콘 회피 모드
current_mode = MODE_LANE_FOLLOWING  # 기본 모드는 차선 추종

# 카메라 보정 관련 변수
camera_cone_detected = False  # 카메라로 라바콘 감지 여부
camera_cone_x = 0  # 카메라로 감지한 라바콘 x 좌표 (화면 중앙 기준)
lane_center = 0  # 차선 중앙 위치 (화면 중앙 기준 -1.0 ~ 1.0) # 새 코드에서는 직접 사용 안함
left_fit = None  # 왼쪽 차선 피팅 계수 # 새 코드에서는 직접 사용 안함
right_fit = None  # 오른쪽 차선 피팅 계수 # 새 코드에서는 직접 사용 안함

# 라이다 시각화 변수
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)  # 시각화 범위 조정 (미터 단위)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo', markersize=2)
obstacle_points, = ax.plot([], [], 'ro', markersize=4)  # 장애물 포인트 (빨간색)
cone_points, = ax.plot([], [], 'yo', markersize=6)  # 라바콘 포인트 (노란색)

# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")
   
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
def lidar_callback(data):
    global ranges    
    ranges = data.ranges[0:360]
    
# 모터로 토픽을 발행하는 함수
def drive(angle, speed):
    motor_msg.angle = float(angle)
    motor_msg.speed = float(speed)
    motor.publish(motor_msg)

# 카메라로 라바콘 색상 인식
def detect_cone_color():
    global camera_cone_detected, camera_cone_x
    
    if image.size == 0:
        camera_cone_detected = False
        return
    
    # 이미지 크기 가져오기
    height, width = image.shape[:2]
    
    # 관심 영역 설정 (하단 절반)
    roi = image[height//2:height, :]
    
    # HSV 색상 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 주황색 필터링
    orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.erode(orange_mask, kernel, iterations=1)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=2)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 라바콘 감지 결과 초기화
    camera_cone_detected = False
    camera_cone_x = 0
    
    if contours:
        # 가장 큰 컨투어 찾기
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 최소 면적 조건
        if cv2.contourArea(largest_contour) > 500:
            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 화면에 표시
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 중심점 계산 (이미지 중앙 기준 좌표)
            center_x = x + w // 2
            cone_x_normalized = (center_x - width // 2) / (width // 2)  # -1.0 ~ 1.0 범위로 정규화
            
            # 라바콘 감지 결과 업데이트
            camera_cone_detected = True
            camera_cone_x = cone_x_normalized
            
            # 정보 표시
            cv2.putText(roi, f"Cone: {cone_x_normalized:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 처리된 영상 표시
    processed_image = image.copy()
    processed_image[height//2:height, :] = cv2.addWeighted(roi, 0.7, cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow("Cone Detection", processed_image)
    cv2.waitKey(1)

# 차선 색상 마스크 생성 (흰색, 노란색 분리)
def create_lane_masks(roi):
    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 흰색 차선 마스크 생성
    white_mask = cv2.inRange(hsv, WHITE_LOWER_NEW, WHITE_UPPER_NEW)
    
    # 노란색 차선 마스크 생성
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER_NEW, YELLOW_UPPER_NEW)
    
    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    white_mask_filtered = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask_filtered = cv2.morphologyEx(white_mask_filtered, cv2.MORPH_CLOSE, kernel)
    
    yellow_mask_filtered = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask_filtered = cv2.morphologyEx(yellow_mask_filtered, cv2.MORPH_CLOSE, kernel)
    
    return white_mask_filtered, yellow_mask_filtered

# 차선 검출 및 추적
def detect_lane():
    global lane_center, left_fit, right_fit
    
    if image.size == 0:
        return
    
    # 이미지 크기 가져오기
    height, width = image.shape[:2]
    
    # ROI 설정
    roi_top = int(height * LANE_ROI_TOP)
    roi_bottom = int(height * LANE_ROI_BOTTOM)
    
    # ROI 추출
    roi = image[roi_top:roi_bottom, :]
    roi_height, roi_width = roi.shape[:2]
    
    # 흰색, 노란색 차선 마스크 생성
    white_lane_mask, yellow_lane_mask = create_lane_masks(roi)
    
    # Canny 엣지 검출 (흰색, 노란색 각각)
    white_edges = cv2.Canny(white_lane_mask, LANE_LOW_THRESHOLD, LANE_HIGH_THRESHOLD)
    yellow_edges = cv2.Canny(yellow_lane_mask, LANE_LOW_THRESHOLD, LANE_HIGH_THRESHOLD)
    
    # 결과 이미지 준비 (컬러로 변경)
    result_image = roi.copy() # 원본 ROI에 직접 그림
    
    # 허프 변환으로 선 감지 (흰색 차선)
    white_lines = cv2.HoughLinesP(
        white_edges, rho=1, theta=np.pi/180, threshold=LANE_HOUGH_THRESHOLD, 
        minLineLength=LANE_MIN_LENGTH, maxLineGap=LANE_MAX_GAP
    )
    
    # 허프 변환으로 선 감지 (노란색 차선 - 중앙선)
    yellow_lines = cv2.HoughLinesP(
        yellow_edges, rho=1, theta=np.pi/180, threshold=LANE_HOUGH_THRESHOLD, 
        minLineLength=LANE_MIN_LENGTH, maxLineGap=LANE_MAX_GAP
    )
    
    # 왼쪽/오른쪽 흰색 차선 및 노란색 중앙선 정보
    left_white_lines = []
    right_white_lines = []
    center_yellow_lines = [] # 중앙선으로 간주
    
    # 흰색 차선 처리
    if white_lines is not None:
        for line in white_lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3: continue
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 20: continue
            
            if slope < 0:  # 왼쪽 흰색 차선
                left_white_lines.append((x1, y1, x2, y2, slope, length))
                cv2.line(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2) # 흰색으로 표시
            else:  # 오른쪽 흰색 차선
                right_white_lines.append((x1, y1, x2, y2, slope, length))
                cv2.line(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2) # 흰색으로 표시

    # 노란색 차선 처리 (중앙선)
    if yellow_lines is not None:
        for line in yellow_lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            # 중앙선은 기울기 범위가 넓을 수 있으므로 약간 완화
            if abs(slope) > 0.2 and abs(slope) < 2.0 : # 너무 수평이거나 수직인 선 제외 
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length < 30: continue # 중앙선은 좀 더 긴 선을 우선
                
                # 화면 중앙 부근에 있는 노란색 선을 중앙선으로 간주
                line_center_x_candidate = (x1 + x2) / 2 # 변수명 변경
                if roi_width * 0.3 < line_center_x_candidate < roi_width * 0.7:
                    center_yellow_lines.append((x1, y1, x2, y2, slope, length))
                    cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2) # 노란색으로 표시

    # 차선 피팅 및 중앙선 계산 로직 수정
    lane_center_x = roi_width // 2  # 기본값
    left_lane_detected = len(left_white_lines) > 0
    right_lane_detected = len(right_white_lines) > 0
    center_lane_detected = len(center_yellow_lines) > 0

    # 가중 평균 계산 함수
    def weighted_average_lane(lines):
        if not lines: return 0, 0, False
        weights = np.array([length for _, _, _, _, _, length in lines])
        total_weight = np.sum(weights)
        if total_weight == 0: return 0, 0, False
        
        # 각 선분의 양 끝점 x, y 좌표를 가져옴
        points_x = []
        points_y = []
        for x1, y1, x2, y2, _, _ in lines:
            points_x.extend([x1, x2])
            points_y.extend([y1, y2])
        
        # 가중 평균 계산 시 각 선분의 가중치를 양 끝점에 동일하게 적용하기 위해 가중치 배열 확장
        expanded_weights = np.repeat(weights, 2)
        
        avg_x = np.sum(np.array(points_x) * expanded_weights) / np.sum(expanded_weights)
        avg_y = np.sum(np.array(points_y) * expanded_weights) / np.sum(expanded_weights)
        return int(avg_x), int(avg_y), True

    left_x, left_y, left_lane_detected = weighted_average_lane(left_white_lines)
    right_x, right_y, right_lane_detected = weighted_average_lane(right_white_lines)
    center_x, center_y, center_lane_detected = weighted_average_lane(center_yellow_lines)

    if left_lane_detected:
        cv2.circle(result_image, (left_x, left_y), 5, (200, 200, 200), -1) # 왼쪽 흰색 차선
    if right_lane_detected:
        cv2.circle(result_image, (right_x, right_y), 5, (200, 200, 200), -1) # 오른쪽 흰색 차선
    if center_lane_detected:
        cv2.circle(result_image, (center_x, center_y), 5, (0, 200, 200), -1) # 노란색 중앙선

    # 주행 경로 결정
    if left_lane_detected and right_lane_detected:
        # 양쪽 흰색 차선이 모두 감지된 경우
        lane_center_x = (left_x + right_x) // 2
    elif left_lane_detected and center_lane_detected:
        # 왼쪽 흰색 차선과 노란색 중앙선이 감지된 경우 (중앙선 왼쪽에 차가 있다고 가정)
        lane_center_x = (left_x + center_x) // 2 
    elif right_lane_detected and center_lane_detected:
        # 오른쪽 흰색 차선과 노란색 중앙선이 감지된 경우 (중앙선 오른쪽에 차가 있다고 가정)
        lane_center_x = (right_x + center_x) // 2 
    elif center_lane_detected: # 중앙선만 감지된 경우
        # 차량은 중앙선의 오른쪽으로 주행한다고 가정 (한국 기준)
        # 이 값은 실제 주행 환경과 규칙에 따라 조정 필요
        lane_center_x = center_x + int(roi_width * 0.15) # 중앙선에서 오른쪽으로 ROI 너비의 15% 만큼 이동 (0.25 -> 0.15)
    elif left_lane_detected:
        # 왼쪽 차선만 감지된 경우
        lane_center_x = left_x + int(roi_width * 0.25) # 왼쪽 차선에서 오른쪽으로 ROI 너비의 25% 만큼 이동 (0.4 -> 0.25)
    elif right_lane_detected:
        # 오른쪽 차선만 감지된 경우
        lane_center_x = right_x - int(roi_width * 0.25) # 오른쪽 차선에서 왼쪽으로 ROI 너비의 25% 만큼 이동 (0.4 -> 0.25)
    else:
        # 차선 감지 실패 시, 이전 값 유지 또는 기본값 사용
        pass # lane_center_x는 roi_width // 2로 유지

    cv2.line(result_image, (lane_center_x, 0), (lane_center_x, roi_height), (0, 0, 255), 2) # 최종 주행 경로 (빨간색)
    
    # 차선 중앙 위치를 -1.0 ~ 1.0 범위로 정규화
    lane_center = (lane_center_x - roi_width / 2) / (roi_width / 2)
    
    # 디버깅 정보 표시
    cv2.putText(result_image, f"L_X: {left_x} R_X: {right_x} C_X: {center_x}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(result_image, f"LaneDetectX: {lane_center_x} Norm: {lane_center:.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(result_image, f"LL:{left_lane_detected} RL:{right_lane_detected} CL:{center_lane_detected}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    # 화면에 표시
    cv2.imshow("Lane Detection", result_image)
    cv2.waitKey(1)

# 라이다 데이터에서 클러스터 찾기 (라바콘 감지용)
def find_clusters(points_x, points_y):
    if len(points_x) == 0:
        return []
    
    # 모든 포인트 페어 간의 거리 계산
    points = list(zip(points_x, points_y))
    clusters = []
    visited = set()
    
    for i, point in enumerate(points):
        if i in visited:
            continue
            
        # 새로운 클러스터 시작
        cluster = [point]
        visited.add(i)
        
        # BFS로 클러스터에 포인트 추가
        queue = [i]
        while queue:
            current = queue.pop(0)
            current_point = points[current]
            
            for j, candidate in enumerate(points):
                if j in visited:
                    continue
                
                # 거리가 임계값보다 작으면 같은 클러스터
                distance = math.sqrt((current_point[0] - candidate[0])**2 + 
                                    (current_point[1] - candidate[1])**2)
                if distance < CONE_CLUSTER_THRESHOLD:
                    cluster.append(candidate)
                    visited.add(j)
                    queue.append(j)
        
        if len(cluster) >= CONE_MIN_POINTS:
            clusters.append(cluster)
    
    return clusters

# 라바콘 감지 함수
def detect_cones():
    if ranges is None:
        return []
    
    angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
    
    # 라바콘 후보 포인트 수집 (적정 거리 내의 포인트)
    candidate_x = []
    candidate_y = []
    for i, r in enumerate(ranges):
        # 무한대 값 제외 및 적정 거리 내의 포인트만 선택
        if not math.isinf(r) and r < OBSTACLE_DISTANCE_THRESHOLD * 3:  # 라바콘 감지는 장애물보다 더 넓은 범위에서
            angle = angles[i]
            candidate_x.append(r * np.cos(angle))
            candidate_y.append(r * np.sin(angle))
    
    # 클러스터 찾기
    clusters = find_clusters(candidate_x, candidate_y)
    
    # 라바콘 후보 클러스터 정제
    cones = []
    for cluster in clusters:
        # 클러스터 중심 계산
        x_coords = [p[0] for p in cluster]
        y_coords = [p[1] for p in cluster]
        center_x_cone = sum(x_coords) / len(x_coords) # 변수명 변경
        center_y_cone = sum(y_coords) / len(y_coords) # 변수명 변경
        
        # 클러스터 높이 추정 (포인트 분산 기반)
        height_estimate = max(
            math.sqrt((max(x_coords) - min(x_coords))**2 + 
                     (max(y_coords) - min(y_coords))**2),
            0.05  # 최소값 설정
        )
        
        # 높이와 기타 특성으로 라바콘 판정
        if CONE_HEIGHT_MIN <= height_estimate <= CONE_HEIGHT_MAX:
            # 중심 좌표, 추정 높이, 포인트 수를 저장
            cones.append({
                'x': center_x_cone,
                'y': center_y_cone,
                'height': height_estimate,
                'points': len(cluster),
                'distance': math.sqrt(center_x_cone**2 + center_y_cone**2)
            })
    
    return cones

# 장애물 감지 함수
def detect_obstacle():
    if ranges is None:
        return False, 0, 0
    
    # 각도별 구간 정의
    front_indices = [(i % 360) for i in range(360-FRONT_ANGLE_RANGE, 360)] + [(i % 360) for i in range(0, FRONT_ANGLE_RANGE+1)]
    left_indices = [(i % 360) for i in range(90-SIDE_ANGLE_RANGE//2, 90+SIDE_ANGLE_RANGE//2+1)]
    right_indices = [(i % 360) for i in range(270-SIDE_ANGLE_RANGE//2, 270+SIDE_ANGLE_RANGE//2+1)]
    
    # 각 구간별 최소 거리 계산
    front_min = min([ranges[i] for i in front_indices if not math.isinf(ranges[i])] or [float('inf')])
    left_min = min([ranges[i] for i in left_indices if not math.isinf(ranges[i])] or [float('inf')])
    right_min = min([ranges[i] for i in right_indices if not math.isinf(ranges[i])] or [float('inf')])
    
    # 전방에 장애물이 있는지 확인
    obstacle_detected = front_min < OBSTACLE_DISTANCE_THRESHOLD
    
    # 회피 방향 결정 (왼쪽과 오른쪽 중 더 넓은 공간이 있는 방향으로)
    avoidance_direction = 1 if left_min > right_min else -1  # 1: 왼쪽, -1: 오른쪽
    
    return obstacle_detected, avoidance_direction, front_min

# 라바콘 회피 전략 결정
def cone_avoidance_strategy(cones):
    global camera_cone_detected, camera_cone_x
    
    if not cones and not camera_cone_detected:
        return 0.0  # 라바콘이 없으면 직진
    
    # 카메라로 라바콘을 감지한 경우 (우선순위 높음)
    if camera_cone_detected:
        # 라바콘이 화면 중앙에서 떨어진 정도에 비례해서 회피 각도 결정
        # 라바콘이 왼쪽에 있으면 오른쪽으로, 오른쪽에 있으면 왼쪽으로 회피
        # 반응성 강화를 위해 계수 증가
        avoidance_angle = camera_cone_x * 50 * CONE_AVOIDANCE_GAIN  # 최대 조향각을 ±50으로 설정, 반응성 증가
        return avoidance_angle
    
    # 카메라 감지가 없을 경우, 라이다 데이터 기반으로 결정
    # 전방 라바콘만 필터링 (y 좌표가 양수)
    front_cones = [cone for cone in cones if cone['y'] > 0]
    if not front_cones:
        return 0.0
    
    # 가장 가까운 라바콘 찾기
    closest_cone = min(front_cones, key=lambda c: c['x']**2 + c['y']**2)
    
    # 라바콘 위치에 따른 회피 방향 결정
    # 라바콘이 왼쪽에 있으면 오른쪽으로, 오른쪽에 있으면 왼쪽으로 회피
    # 반응성 강화를 위해 계수 증가
    angle_rad = math.atan2(closest_cone['y'], closest_cone['x'])
    distance_factor = 1.0 / max(closest_cone['distance'], 0.3)  # 거리가 가까울수록 회피 각도 증가
    avoidance_angle = -math.degrees(angle_rad) * CONE_AVOIDANCE_GAIN * distance_factor
    
    # 회피 각도 제한 (-50 ~ 50)
    avoidance_angle = max(-50, min(50, avoidance_angle))
    
    return avoidance_angle

# 차선 추종 제어 - 새로운 코드로 대체
def lane_following_control():
    global image, frame_count, prev_angle, white_lost_count # 필요한 전역 변수 사용 명시

    if image is None or image.size == 0:
        return 0.0, DEFAULT_SPEED # 이미지가 없으면 기본값 반환

    height, width = image.shape[:2]
    roi_img = image[int(height * 0.6):, :] # 새 코드의 ROI 설정
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # 새 코드의 흰색/노란색 마스크 생성
    white_mask = cv2.inRange(hsv, WHITE_LOWER_NEW, WHITE_UPPER_NEW)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER_NEW, YELLOW_UPPER_NEW)

    frame_count += 1
    angle = 0.0
    speed = DEFAULT_SPEED
    direction = "INIT"
    
    # 오류 해결을 위해 변수 초기화
    left_ratio = 0.0
    mid_ratio = 0.0
    right_ratio = 0.0
    total_white = 0

    if frame_count <= 50:
        angle = 0
        speed = 60 # 초기 주행 속도
        direction = f"[INIT] Frame {frame_count} → 직진"
    else:
        left_mask_roi  = white_mask[:, :width//3]
        # mid_mask_roi   = white_mask[:, width//3:2*width//3] # roi 이미지 기준이므로 width는 roi_img.shape[1] 사용해야함
        # right_mask_roi = white_mask[:, 2*width//3:]
        roi_width = roi_img.shape[1]
        left_mask_roi  = white_mask[:, :roi_width//3]
        mid_mask_roi   = white_mask[:, roi_width//3:(2*roi_width)//3]
        right_mask_roi = white_mask[:, (2*roi_width)//3:]

        left_ratio  = cv2.countNonZero(left_mask_roi)  / (left_mask_roi.size if left_mask_roi.size > 0 else 1)
        mid_ratio   = cv2.countNonZero(mid_mask_roi)   / (mid_mask_roi.size if mid_mask_roi.size > 0 else 1) # 분모 0 방지
        right_ratio = cv2.countNonZero(right_mask_roi) / (right_mask_roi.size if right_mask_roi.size > 0 else 1)
        total_white = cv2.countNonZero(white_mask)

        if total_white > 300:
            white_lost_count = 0
            error = (left_ratio - right_ratio) * 100
            angle = np.clip(error * 0.6, -30, 30)

            if left_ratio - right_ratio > 0.05:
                angle += 15
            elif right_ratio - left_ratio > 0.05:
                angle -= 15
            angle = np.clip(angle, -50, 50) # 추가 조향 후에도 최대값 제한
            direction = "WHITE_TRACK"
        else:
            white_lost_count += 1
            if white_lost_count >= 1:  # fallback 바로 진입
                M = cv2.moments(yellow_mask)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    error_yellow = cx - (roi_width // 2)
                    angle = np.clip(error_yellow * 0.005, -25, 25) # 새 코드의 계수 및 범위
                    direction = "YELLOW_FALLBACK"
                else:
                    angle = -15  # 기본 회피각 부여
                    direction = "NO_LINE_ESCAPE"
            else:
                # 이 경우는 white_lost_count가 0일 때만 오므로, 실제로는 거의 발생 안 함 (위에서 white_lost_count = 0으로 초기화)
                # 만약 white_lost_count 조건이 >= 1 이 아니라 > N (N > 0) 이었다면 의미 있을 수 있음.
                # 현재 로직에서는 white_lost_count가 0이면 흰색 차선 추종을 시도함.
                angle = prev_angle 
                direction = f"WHITE_LOST_{white_lost_count}"

        # angle 변화 제한
        max_delta = 10
        if abs(angle - prev_angle) > max_delta:
            angle = prev_angle + np.sign(angle - prev_angle) * max_delta
        
        prev_angle = angle

        # 속도 결정
        abs_angle = abs(angle)
        if "FALLBACK" in direction or "NO_LINE" in direction : # NO_LINE_ESCAPE 포함
             speed = 30        
        elif abs_angle < 5:
            speed = 80
        elif abs_angle < 10:
            speed = 60
        else:
            speed = 45
       
    # 새 코드의 시각화 로직
    # 원본 이미지에 ROI 영역 표시 추가
    debug_image = image.copy()
    cv2.rectangle(debug_image, (0, int(height*0.6)), (width-1, height-1), (0,255,0), 2)
    cv2.putText(debug_image, f"Mode: {direction}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
    cv2.putText(debug_image, f"Angle:{angle:.1f} Speed:{speed}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
    cv2.putText(debug_image, f"LMR: {left_ratio:.2f}|{mid_ratio:.2f}|{right_ratio:.2f} TW:{total_white}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2) 
    
    cv2.imshow("Robust Lane Following", debug_image)
    cv2.imshow("White Mask ROI", white_mask)
    cv2.imshow("Yellow Mask ROI", yellow_mask)
    cv2.waitKey(1)
    
    return angle, speed

# 주행 모드 결정 (장애물 회피, 라바콘 회피, 차선 추종)
def determine_driving_mode():
    global current_mode
    
    # 라이다로 장애물 감지
    obstacle_detected, _, _ = detect_obstacle() # distance 변수 사용 안함
    
    # 라바콘 감지
    cones = detect_cones()
    closest_cone_dist = min([cone['distance'] for cone in cones] if cones else [float('inf')])
    
    # 라바콘이 카메라로 감지되었거나 라이다로 감지된 경우
    if camera_cone_detected or closest_cone_dist < OBSTACLE_DISTANCE_THRESHOLD * 1.5: # 라바콘 감지 거리 임계값 조정
        current_mode = MODE_CONE_AVOIDANCE
    # 일반 장애물이 감지된 경우
    elif obstacle_detected:
        current_mode = MODE_OBSTACLE_AVOIDANCE
    # 그 외의 경우는 차선 추종
    else:
        current_mode = MODE_LANE_FOLLOWING
    
    return current_mode

# 장애물 및 라바콘 회피 알고리즘 통합
def obstacle_avoidance():
    # 일반 장애물 감지
    obstacle_detected, avoidance_direction, distance = detect_obstacle()
    
    # 라바콘 감지 (라이다)
    cones = detect_cones()
    
    # 라바콘이 있는 경우 라바콘 회피 우선
    if cones or camera_cone_detected:
        cone_angle = cone_avoidance_strategy(cones)
        if obstacle_detected:
            # 일반 장애물과 라바콘이 모두 있는 경우, 더 가까운 쪽을 회피
            if distance < min([cone['distance'] for cone in cones] if cones else [float('inf')]):
                # 장애물과의 거리에 따라 속도 조절
                speed = max(DEFAULT_SPEED * 0.7, 5)  # 최소 속도 더 낮춤
                
                # 회피 방향으로 조향각 설정 (더 강한 회피 반응)
                avoidance_angle = avoidance_direction * 50  # 최대 회전각 적용
                
                return avoidance_angle, speed
            else:
                # 라바콘 회피
                return cone_angle, DEFAULT_SPEED * 0.8
        else:
            # 라바콘만 회피
            return cone_angle, DEFAULT_SPEED * 0.9
    elif obstacle_detected:
        # 일반 장애물만 회피
        speed = max(DEFAULT_SPEED * 0.7, 5)  # 최소 속도 더 낮춤
        
        # 회피 방향으로 조향각 설정 (더 강한 반응)
        avoidance_angle = avoidance_direction * 50
        
        return avoidance_angle, speed
    else:
        # 장애물이 없으면 직진 (기본 속도)
        return 0.0, DEFAULT_SPEED

# 통합 주행 제어 (Turoad/lanedet 참고)
def drive_control():
    # 주행 모드 결정
    mode = determine_driving_mode()
    global image # lane_following_control이 image를 직접 사용하므로
    
    angle = 0.0
    speed = DEFAULT_SPEED

    if mode == MODE_CONE_AVOIDANCE or mode == MODE_OBSTACLE_AVOIDANCE:
        # 장애물 또는 라바콘 회피 모드
        angle, speed = obstacle_avoidance()
        # 장애물/라바콘 회피 시에는 새 차선 로직의 디버깅 창을 닫거나 다른 내용 표시
        # 여기서는 일단 열려있는 창들을 그대로 두거나, 필요시 cv2.destroyWindow() 등으로 특정 창만 닫을 수 있음
    else: # MODE_LANE_FOLLOWING
        # 차선 추종 모드
        if image is not None and image.size > 0: # 이미지가 있을 때만 차선 추종
            angle, speed = lane_following_control() 
        else:
            angle = 0.0 # 이미지가 없으면 직진
            speed = 0 # 또는 정지
    
    # 모드에 따른 정보 표시 (기존 로그)
    mode_text = "차선 추종" if mode == MODE_LANE_FOLLOWING else "장애물 회피" if mode == MODE_OBSTACLE_AVOIDANCE else "라바콘 회피"
    # 새로운 lane_center 값은 없으므로 로그에서 제거하거나, prev_angle 등으로 대체 가능
    # print(f"모드: {mode_text}, 각도: {angle:.1f}, 속도: {speed:.1f}, LaneCenter: {lane_center:.2f}") 
    print(f"모드: {mode_text}, 최종 각도: {angle:.1f}, 최종 속도: {speed:.1f}, PrevAngle: {prev_angle:.2f}")
    
    return angle, speed

# 라이다 데이터 및 감지된 객체 시각화
def visualize_lidar():
    if ranges is None:
        return
        
    angles = np.linspace(0, 2*np.pi, len(ranges)) + np.pi/2
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    # 모든 라이다 포인트 시각화
    lidar_points.set_data(x, y)
    
    # 장애물 포인트 시각화 (임계값보다 가까운 포인트)
    obstacle_x_viz = [] # 변수명 변경
    obstacle_y_viz = [] # 변수명 변경
    for i, r in enumerate(ranges):
        if not math.isinf(r) and r < OBSTACLE_DISTANCE_THRESHOLD:
            angle_rad_viz = angles[i] # 변수명 변경
            obstacle_x_viz.append(r * np.cos(angle_rad_viz))
            obstacle_y_viz.append(r * np.sin(angle_rad_viz))
    
    obstacle_points.set_data(obstacle_x_viz, obstacle_y_viz)
    
    # 라바콘 시각화
    cones_viz = detect_cones() # 변수명 변경
    cone_x_viz = [cone['x'] for cone in cones_viz]
    cone_y_viz = [cone['y'] for cone in cones_viz]
    cone_points.set_data(cone_x_viz, cone_y_viz)
    
    # 주행 모드에 따라 제목 설정
    mode_text = "차선 추종" if current_mode == MODE_LANE_FOLLOWING else "장애물 회피" if current_mode == MODE_OBSTACLE_AVOIDANCE else "라바콘 회피"
    plt.title(f"모드: {mode_text}, CamCone: {'OK' if camera_cone_detected else 'No'}, LC: {lane_center:.2f}") # LaneCenter 값 시각화 추가
    
    fig.canvas.draw_idle()
    plt.pause(0.01)

# 실질적인 메인 함수
def start():
    global motor, image, ranges
    
    print("라이다 장애물 회피 주행 시작 --------------")

    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다
    rospy.init_node('Lidar_Obstacle_Avoidance')
    rospy.Subscriber("/usb_cam/image_raw/", Image, usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
        
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다
    rospy.wait_for_message("/scan", LaserScan)
    print("라이다 준비 완료 ----------")
    
    # 카메라 토픽이 있으면 기다립니다
    try:
        rospy.wait_for_message("/usb_cam/image_raw/", Image, timeout=3)
        print("카메라 준비 완료 --------------")
        camera_enabled = True
    except rospy.ROSException:
        print("카메라가 연결되지 않았습니다. 라이다만 사용합니다.")
        camera_enabled = False
    
    # 라이다 시각화 준비
    plt.ion()
    plt.show()
    print("라이다 시각화 준비 완료 ----------")
    
    print("======================================")
    print(" 자 율 주 행   시 작 . . . ")
    print("======================================")

    # 메인 루프
    rate = rospy.Rate(10)  # 10Hz로 실행
    while not rospy.is_shutdown():
        # 카메라로 라바콘 색상 감지 및 차선 감지
        if camera_enabled and image.size != 0:
            detect_cone_color()
            # detect_lane()  # 기존 차선 감지 함수 호출은 제거 또는 조건부 실행
            # 차선 추종 모드가 아닐 때만 기존 detect_lane을 호출하거나, 아예 호출하지 않음.
            # 새로운 lane_following_control 함수가 차선 추종 시 이미지 처리를 담당.
            if current_mode != MODE_LANE_FOLLOWING:
                # 만약 다른 모드에서 레거시 차선 정보(lane_center 등)가 필요하다면 여기서 detect_lane() 호출
                # 하지만 현재 obstacle_avoidance 등은 lane_center를 사용하지 않음.
                # detect_lane() # 일단 주석 처리
                pass # 다른 모드에서 기존 차선 감지가 필요하면 여기에 추가
        
        # 주행 제어 (차선 추종 + 장애물/라바콘 회피)
        # drive_control 내부에서 current_mode를 참조하여 lane_following_control을 호출함.
        angle_to_drive, speed_to_drive = drive_control()
        
        # 모터 제어
        drive(angle=angle_to_drive, speed=speed_to_drive)
        
        # 라이다 시각화
        visualize_lidar() # 라이다 시각화는 계속 수행
        
        rate.sleep()
    
    # 종료 처리
    plt.close('all')
    cv2.destroyAllWindows()

# 메인함수 호출
if __name__ == '__main__':
    try:
        start()
    except rospy.ROSInterruptException:
        pass 