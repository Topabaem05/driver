#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class ObstacleAvoider:
    def __init__(self):
        rospy.init_node('obstacle_avoider_node', anonymous=True)

        # 매개변수 (필요에 따라 조정하세요)
        self.obstacle_threshold_front = 0.5  # 전방 장애물 감지 거리 (m)
        self.obstacle_threshold_side = 0.3   # 측면 장애물 감지 거리 (회전 결정시)
        self.safe_turn_distance = 0.7        # 회전하기에 안전하다고 판단하는 측면 거리 (m)

        self.forward_speed = 0.15            # 직진 속도 (m/s)
        self.turn_speed = 0.4                # 회전 속도 (rad/s)

        # Publisher & Subscriber
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.twist_msg = Twist()
        self.lidar_ranges = None
        self.num_lidar_points = None # 라이다 포인트 수 (scan_callback에서 초기화)
        
        # 라이다 센서의 각도 범위에 따라 영역을 정의하기 위한 변수
        # 보통 TurtleBot3의 경우 360개의 포인트가 0도(전방)에서 시작하여 반시계 방향으로 증가합니다.
        # 전방: 0도 부근 / 좌측: 30도~90도 / 우측: 270도~330도 (-90도~-30도)
        # 아래 값들은 전체 라이다 포인트 수를 기준으로 비율로 계산됩니다.
        self.front_cone_deg = 30  # 전방을 판단할 좌우 각도 (예: +/- 30도 = 총 60도)
        self.side_cone_deg_start = 30 # 측면 판단 시작 각도 (전방으로부터)
        self.side_cone_deg_end = 90   # 측면 판단 종료 각도 (전방으로부터)

        rospy.loginfo("Obstacle Avoider Node Initialized")
        rospy.loginfo(f"Params: Front_Thresh={self.obstacle_threshold_front}, Side_Thresh={self.obstacle_threshold_side}")
        rospy.loginfo(f"Speed: Forward={self.forward_speed}, Turn={self.turn_speed}")


    def scan_callback(self, data):
        """
        LaserScan 데이터를 받아서 처리하고 로봇을 움직입니다.
        """
        if self.num_lidar_points is None:
            self.num_lidar_points = len(data.ranges)
            if self.num_lidar_points == 0:
                rospy.logwarn_throttle(5, "Lidar data has 0 points. Obstacle avoidance might not work.")
                return
            rospy.loginfo(f"Lidar detected {self.num_lidar_points} points.")


        # inf, nan 값을 최대 감지 거리로 대체 (또는 매우 큰 값)
        self.lidar_ranges = np.array(data.ranges)
        self.lidar_ranges[np.isinf(self.lidar_ranges)] = data.range_max
        self.lidar_ranges[np.isnan(self.lidar_ranges)] = data.range_max


        # --- 영역별 최소 거리 계산 ---
        # 주의: 이 부분은 라이다 센서의 마운트 방향과 angle_min, angle_max에 따라 달라질 수 있습니다.
        # 아래 코드는 일반적인 TurtleBot3의 라이다 (0도가 전방, 반시계 방향으로 증가, 360개 포인트)를 가정합니다.
        
        # 전방 영역 인덱스 계산 (0도 기준 +/- self.front_cone_deg/2)
        # num_lidar_points가 360일때, front_cone_deg가 30이면, idx_cone_half는 15.
        # 전방: 0~14, 345~359
        idx_cone_half = int((float(self.front_cone_deg) / 360.0) * self.num_lidar_points / 2.0)
        
        # 좌측 영역 인덱스 계산 (예: 30도 ~ 90도)
        idx_side_start = int((float(self.side_cone_deg_start) / 360.0) * self.num_lidar_points)
        idx_side_end = int((float(self.side_cone_deg_end) / 360.0) * self.num_lidar_points)

        # 전방 영역 거리값 추출
        # np.concatenate를 사용하여 배열의 시작과 끝 부분을 합칩니다 (0도 주변)
        if idx_cone_half > 0:
             ranges_front = np.concatenate((
                self.lidar_ranges[0:idx_cone_half+1],
                self.lidar_ranges[self.num_lidar_points - idx_cone_half : self.num_lidar_points]
            ))
        else: # 만약 cone이 너무 작으면 정면만
            ranges_front = np.array([self.lidar_ranges[0]])


        # 좌측 영역 거리값 추출
        ranges_left = self.lidar_ranges[idx_side_start : idx_side_end]
        
        # 우측 영역 거리값 추출 (좌측과 대칭)
        ranges_right = self.lidar_ranges[self.num_lidar_points - idx_side_end : self.num_lidar_points - idx_side_start]

        min_front_dist = np.min(ranges_front) if len(ranges_front) > 0 else float('inf')
        min_left_dist = np.min(ranges_left) if len(ranges_left) > 0 else float('inf')
        min_right_dist = np.min(ranges_right) if len(ranges_right) > 0 else float('inf')

        rospy.logdebug(f"Distances: Front={min_front_dist:.2f}m, Left={min_left_dist:.2f}m, Right={min_right_dist:.2f}m")

        # --- 이동 결정 로직 ---
        linear_x = 0.0
        angular_z = 0.0

        if min_front_dist > self.obstacle_threshold_front:
            # 전방이 안전하면 직진
            linear_x = self.forward_speed
            rospy.logdebug("Path clear, moving forward.")
        else:
            # 전방에 장애물 감지, 정지 후 회전
            linear_x = 0.0
            rospy.loginfo(f"Obstacle detected in front ({min_front_dist:.2f}m). Turning...")
            
            # 좌측과 우측 중 더 안전하고 넓은 곳으로 회전
            # (측면 자체에도 장애물이 없는지 확인 - obstacle_threshold_side)
            can_turn_left = min_left_dist > self.obstacle_threshold_side
            can_turn_right = min_right_dist > self.obstacle_threshold_side

            if can_turn_left and (min_left_dist > min_right_dist or not can_turn_right):
                angular_z = self.turn_speed  # 좌회전
                rospy.logdebug("Turning left.")
            elif can_turn_right and (min_right_dist > min_left_dist or not can_turn_left):
                angular_z = -self.turn_speed # 우회전
                rospy.logdebug("Turning right.")
            else: # 양쪽 다 비슷하게 막혔거나, 회전하기 위험한 경우
                # 기본적으로 좌회전 (또는 후진 등 다른 전략 추가 가능)
                angular_z = self.turn_speed 
                rospy.logdebug("Both sides blocked or risky, defaulting to left turn.")
        
        self.twist_msg.linear.x = linear_x
        self.twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(self.twist_msg)

    def run(self):
        rospy.spin() # 콜백 함수가 메시지를 계속 처리하도록 유지

if __name__ == '__main__':
    try:
        avoider = ObstacleAvoider()
        avoider.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Obstacle Avoider Node Shutting Down.")
    finally:
        # 노드 종료 시 로봇을 정지시키는 것이 좋습니다.
        final_twist = Twist()
        final_twist.linear.x = 0
        final_twist.angular.z = 0
        # Publisher가 아직 살아있을 수 있으므로 try-except로 감쌉니다.
        try:
            if hasattr(avoider, 'cmd_vel_pub'): # avoider 객체가 생성되었고, publisher가 있다면
                 avoider.cmd_vel_pub.publish(final_twist)
                 rospy.loginfo("Published zero velocity.")
        except Exception as e:
            rospy.logerr(f"Error publishing zero velocity on shutdown: {e}")