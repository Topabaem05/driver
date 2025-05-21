#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import math
import pygame
from sensor_msgs.msg import LaserScan

# --- Pygame 설정 ---
pygame.init()
TOTAL_SCREEN_WIDTH = 1200  # 전체 화면 너비 (두 뷰어를 위해 넓힘)
SCREEN_HEIGHT = 600    # 전체 화면 높이

# 화면 분할
TOP_VIEW_WIDTH = TOTAL_SCREEN_WIDTH // 2
FRONT_VIEW_WIDTH = TOTAL_SCREEN_WIDTH // 2
VIEW_HEIGHT = SCREEN_HEIGHT # 높이는 동일하게 사용

screen = pygame.display.set_mode((TOTAL_SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LiDAR Multi-View Visualizer")

# 각 뷰어에 대한 Surface 생성
top_view_surface = pygame.Surface((TOP_VIEW_WIDTH, VIEW_HEIGHT))
front_view_surface = pygame.Surface((FRONT_VIEW_WIDTH, VIEW_HEIGHT))

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)

# --- Top-View 시각화 설정 ---
T_PIXELS_PER_METER = 50  # Top-View: 1미터를 몇 픽셀로 (화면 크기에 맞게 조절)
T_MAX_DISPLAY_RANGE_METERS = 50.0 # Top-View: 화면에 표시할 최대 거리 (미터)
T_ROBOT_SIZE_PIXELS = 10 # Top-View: 로봇 사각형의 한 변 길이의 절반 (픽셀)
T_POINT_RADIUS_PIXELS = 1   # Top-View: LiDAR 포인트의 반지름 (픽셀) - 정밀도 향상

# --- Front-View 시각화 설정 ---
F_FOV_DEGREES = 180  # Front-View: 전방 시야각 (좌우 대칭)
F_MAX_BAR_HEIGHT_PIXELS = VIEW_HEIGHT - 50 # Front-View: 막대 최대 높이 (여유 공간)
F_MAX_DISTANCE_METERS = 6.0 # Front-View: 이 거리에서 막대가 최대 높이가 됨 (T_MAX_DISPLAY_RANGE_METERS와 유사하게 설정)
F_BAR_COLOR = YELLOW

# --- ROS 관련 설정 ---
ranges_data = None # 전체 LiDAR 데이터 (0~359도)
lidar_ready = False

def lidar_callback(data):
    global ranges_data, lidar_ready
    # data.ranges는 일반적으로 0도(로봇 전방)부터 반시계 방향으로 측정됨
    # 대부분의 LiDAR는 360개의 값을 가짐. angle_min, angle_increment를 사용하는 것이 더 일반적.
    # 여기서는 360개의 데이터 포인트가 있다고 가정하고 사용.
    # 실제 환경에서는 data.angle_min, data.angle_increment, len(data.ranges)를 확인해야 함.
    num_points = len(data.ranges)
    if num_points > 0:
        # 여기서는 360개로 고정된 슬라이싱 대신 전체 데이터를 받음
        # 이후 각 뷰어에서 필요한 부분을 처리
        ranges_data = list(data.ranges) 
    
    if not lidar_ready and ranges_data:
        lidar_ready = True
        print("Lidar Data Received. Visualizer Ready ----------")

def draw_top_view(surface, current_ranges):
    surface.fill(BLACK) # 배경색
    
    center_x = TOP_VIEW_WIDTH // 2
    center_y = VIEW_HEIGHT // 2

    # 로봇(중심) 그리기 - 빨간색 사각형
    robot_rect = pygame.Rect(center_x - T_ROBOT_SIZE_PIXELS, 
                             center_y - T_ROBOT_SIZE_PIXELS, 
                             T_ROBOT_SIZE_PIXELS * 2, 
                             T_ROBOT_SIZE_PIXELS * 2)
    pygame.draw.rect(surface, RED, robot_rect)
    
    if not current_ranges:
        return

    num_scan_points = len(current_ranges)
    if num_scan_points == 0:
        return

    for i in range(num_scan_points):
        # 가정: ranges_data[0]가 0도(전방), 시계 반대 방향으로 각도 증가
        # 각도는 인덱스 i와 같다고 가정 (0 ~ num_scan_points-1)
        # 더 정확하게는 LiDAR의 angle_min, angle_increment 사용 필요
        angle_degrees = (360.0 / num_scan_points) * i 
        distance_meters = current_ranges[i]

        if math.isinf(distance_meters) or math.isnan(distance_meters) or distance_meters == 0.0:
            continue
        if distance_meters > T_MAX_DISPLAY_RANGE_METERS:
            continue
        
        angle_radians = math.radians(angle_degrees)
        
        x_robot = distance_meters * math.cos(angle_radians)
        y_robot = distance_meters * math.sin(angle_radians)

        screen_x = center_x + int(x_robot * T_PIXELS_PER_METER)
        screen_y = center_y - int(y_robot * T_PIXELS_PER_METER) # y축 반전

        if 0 <= screen_x < TOP_VIEW_WIDTH and 0 <= screen_y < VIEW_HEIGHT:
             pygame.draw.circle(surface, GREEN, (screen_x, screen_y), T_POINT_RADIUS_PIXELS)
    
    # 뷰어 제목
    font = pygame.font.Font(None, 24)
    text = font.render("Top View", 1, WHITE)
    surface.blit(text, (10, 10))


def draw_front_view(surface, current_ranges):
    surface.fill(GRAY) # 배경색

    if not current_ranges:
        return

    num_total_points = len(current_ranges)
    if num_total_points == 0:
        return

    # 전방 시야각에 해당하는 데이터 추출
    # 가정: ranges_data[0]이 0도(전방), 반시계 방향. num_total_points가 360이라고 가정.
    # 예: 360개 데이터 포인트 기준. 실제로는 angle_min/max/increment 사용해야 함.
    # 각도당 포인트 수: num_total_points / 360.0
    
    # 전방 FOV/2 에 해당하는 포인트 수 계산
    # (F_FOV_DEGREES / 2) 도 만큼의 데이터 포인트
    # angle_increment_deg = 360.0 / num_total_points
    # points_per_side = int(round((F_FOV_DEGREES / 2.0) / angle_increment_deg))
    
    # 단순화: ranges_data가 360개이고 1도 간격이라고 강하게 가정
    # 실제로는 angle_min, angle_max, angle_increment를 사용해 정확한 인덱스 계산 필요
    # 0도(전방) 인덱스가 0이라고 가정
    # 오른쪽 시야: (360 - FOV/2)도 부터 359도까지
    # 왼쪽 시야: 0도 부터 (FOV/2)도 까지

    # 이 예제에서는 ranges_data가 0도부터 시작하여 360개의 값을 가진다고 가정합니다.
    # 실제 LiDAR에 맞게 이 부분을 조정해야 합니다. (e.g. RPLidar는 0도가 뒤쪽일 수 있음)
    # 현재 코드는 ranges[0]가 전방, 반시계방향으로 증가한다고 가정.
    
    # 데이터 슬라이싱 (360개 데이터 기준, 0도가 전방)
    center_angle_deg = 0 
    half_fov_deg = F_FOV_DEGREES / 2.0

    # 오른쪽 시야 (음의 각도, 데이터 배열에서는 뒷부분)
    # 예: -90도 ~ -1도  ==> 270도 ~ 359도
    start_idx_right = int(round((360.0 - half_fov_deg) / (360.0 / num_total_points))) % num_total_points
    end_idx_right = num_total_points # (실제로는 num_total_points -1 까지)

    # 왼쪽 시야 (양의 각도, 데이터 배열에서는 앞부분)
    # 예: 0도 ~ 90도
    start_idx_left = 0
    end_idx_left = int(round(half_fov_deg / (360.0 / num_total_points))) + 1
    
    # 전방 시야 데이터 (오른쪽 -> 전방 -> 왼쪽 순서로 정렬)
    # 주의: LiDAR가 0도를 어디로 잡느냐에 따라 이 인덱싱은 크게 달라짐
    # 이 코드는 ranges[0]이 정확히 로봇의 전방 0도라고 가정합니다.
    # 만약 ranges가 360개라면:
    if num_total_points == 360: # 가장 일반적인 케이스
        # 오른쪽 90도 (270~359), 왼쪽 90도 (0~90) for 180도 FOV
        front_view_scan_data = current_ranges[270:360] + current_ranges[0:91]
    else: # 일반화 시도 (하지만 angle_min, angle_increment 정보가 없어 부정확할 수 있음)
        # 이 부분은 LiDAR 스펙에 맞춰야 합니다.
        # 간단히 중앙 부분의 F_FOV_DEGREES 만큼을 사용하도록 할 수 있지만, 0도 정렬이 중요.
        # 여기서는 360개 데이터가 아니면 앞부분 F_FOV_DEGREES 만큼만 사용 (단순화)
        num_points_to_show = int(round(F_FOV_DEGREES * (num_total_points / 360.0)))
        # 0도(전방)가 중앙에 오도록 데이터를 회전시켜야 함.
        # 여기서는 current_ranges[0]이 전방이라는 가정 하에, 앞부분부터 사용
        # front_view_scan_data = current_ranges[:num_points_to_show] # 이것은 올바른 FOV 표현이 아님.
        # 임시로 전체 데이터의 앞부분을 사용 (실제론 이렇게 하면 안됨)
        print(f"Warning: num_total_points is {num_total_points}, not 360. Front view may be incorrect.")
        points_for_fov = int(F_FOV_DEGREES * (num_total_points / 360.0))
        # 0도를 중심으로 points_for_fov 만큼을 가져오는 로직 필요
        # 여기서는 데모를 위해 단순화: 앞 부분 절반 + 뒷 부분 절반 (0도가 중앙에 오도록)
        half_points_fov = points_for_fov // 2
        front_view_scan_data = current_ranges[-half_points_fov:] + current_ranges[:half_points_fov+1]


    if not front_view_scan_data:
        return

    num_front_points = len(front_view_scan_data)
    if num_front_points == 0:
        return

    bar_width = float(FRONT_VIEW_WIDTH) / num_front_points
    
    for i in range(num_front_points):
        distance_meters = front_view_scan_data[i]

        if math.isinf(distance_meters) or math.isnan(distance_meters) or distance_meters == 0.0:
            bar_height = 0
        elif distance_meters > F_MAX_DISTANCE_METERS:
            bar_height = F_MAX_BAR_HEIGHT_PIXELS # 최대 거리 초과 시 최대 높이
        else:
            # 거리에 비례하여 막대 높이 계산
            bar_height = int((distance_meters / F_MAX_DISTANCE_METERS) * F_MAX_BAR_HEIGHT_PIXELS)
            if bar_height < 0: bar_height = 0 # 음수 방지
        
        # 막대 그리기 (아래에서 위로)
        bar_x = i * bar_width
        # pygame y좌표는 위에서 아래로 증가. 막대는 화면 바닥에서 시작.
        bar_y = VIEW_HEIGHT - bar_height 
        
        pygame.draw.rect(surface, F_BAR_COLOR, (bar_x, bar_y, math.ceil(bar_width), bar_height))

    # 뷰어 제목
    font = pygame.font.Font(None, 24)
    text = font.render("Front View (180 deg)", 1, BLACK) # 배경이 밝으니 글자색 어둡게
    surface.blit(text, (10, 10))
    
    # 중앙선 (전방 0도 표시)
    center_line_x = FRONT_VIEW_WIDTH // 2
    pygame.draw.line(surface, RED, (center_line_x, VIEW_HEIGHT - 20), (center_line_x, VIEW_HEIGHT), 2)
    text_0_deg = font.render("0°", 1, RED)
    surface.blit(text_0_deg, (center_line_x - text_0_deg.get_width()//2 , VIEW_HEIGHT - 40))


def main():
    global ranges_data # 전역 변수 사용 명시

    rospy.init_node('Lidar_Multi_Visualizer', anonymous=True)
    # queue_size=1로 하여 항상 최신 데이터만 처리
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1) 

    print("Waiting for LiDAR data...")

    clock = pygame.time.Clock()
    running = True

    while running and not rospy.is_shutdown():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not lidar_ready or ranges_data is None:
            # 데이터 로딩 중 표시 (전체 화면)
            screen.fill(BLACK)
            font = pygame.font.Font(None, 48)
            text = font.render("Waiting for LiDAR data...", 1, WHITE)
            text_rect = text.get_rect(center=(TOTAL_SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(10) # 데이터 없을 땐 느리게
            continue

        # 각 뷰어 그리기
        # 현재 ranges_data를 복사해서 전달하여 그리는 도중 변경되는 것을 방지 (선택적)
        current_scan_data = list(ranges_data) if ranges_data else []

        draw_top_view(top_view_surface, current_scan_data)
        draw_front_view(front_view_surface, current_scan_data)

        # 메인 화면에 각 뷰어 Surface를 blit
        screen.blit(top_view_surface, (0, 0))
        screen.blit(front_view_surface, (TOP_VIEW_WIDTH, 0))
        
        # 화면 업데이트
        pygame.display.flip()
        clock.tick(30) # 초당 프레임 제한

    pygame.quit()
    print("Visualizer stopped.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.quit()