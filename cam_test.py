#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, numpy as np, os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()
cv_image = np.empty(shape=[0])

# OpenCV GUI 환경 변수 설정 (리눅스 환경에서 도움될 수 있음)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def detect_traffic_light(image):
    # 이미지 디버깅 정보 출력
    print(f"입력 이미지 크기: {image.shape}, 타입: {image.dtype}, 비어있음: {image.size == 0}")
    
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        print("경고: 유효하지 않은 이미지 입력")
        return None, None, None, None, None, None, None
        
    height, width = image.shape[:2]
    
    # 상단 부분만 ROI(Region of Interest)로 설정 - 상단 40%만 사용
    roi_height = int(height * 0.4)  # 상단 40%
    roi = image[0:roi_height, 0:width]
    
    # ROI 표시용 이미지
    roi_display = image.copy()
    # ROI 영역 표시 (녹색 사각형)
    cv2.rectangle(roi_display, (0, 0), (width, roi_height), (0, 255, 0), 2)
    
    # 이미지 전처리 - 약간의 흐림 효과로 노이즈 감소
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # HSV 이미지로 변환 (ROI 영역만)
    hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 (빨간색은 HSV에서 두 범위로 나뉨)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 노란색 범위 (확장된 범위)
    lower_yellow = np.array([15, 70, 70])   # H 값 하한 낮춤, S와 V 값도 낮춤
    upper_yellow = np.array([35, 255, 255]) # H 값 상한 높임
    
    # 초록색 범위 (범위 확장)
    # 초록색 HSV 범위를 넓게 조정 (H: 35-90, S: 50-255, V: 50-255)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # 각 색상 마스크 생성
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 노란색 마스크 생성 및 형태학적 연산 적용
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 노란색 마스크에 모폴로지 연산 적용 (노이즈 제거 및 검출 영역 강화)
    kernel = np.ones((5, 5), np.uint8)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)  # 작은 노이즈 제거
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_DILATE, kernel)  # 영역 확장
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 각 색상의 픽셀 수 계산
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    
    # 결과 이미지 생성
    result_image = image.copy()
    
    # 픽셀 임계값 (조정 가능)
    threshold_red = 300
    threshold_yellow = 200  # 노란색 임계값 더 낮게 설정
    threshold_green = 200
    
    # 픽셀 수 디버깅 정보 출력 (특히 노란색 강조)
    print(f"ROI 내 감지된 픽셀 수: 빨간색={red_pixels}, 노란색={yellow_pixels} (임계값: {threshold_yellow}), 초록색={green_pixels}")
    
    # 감지된 색상 정보 표시
    text = "감지된 색상: "
    if red_pixels > threshold_red:
        text += "빨간색 (정지)"
        cv2.putText(result_image, "RED - STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif yellow_pixels > threshold_yellow:
        text += "노란색 (정지)"
        cv2.putText(result_image, "YELLOW - STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif green_pixels > threshold_green:
        text += "초록색 (출발)"
        cv2.putText(result_image, "GREEN - GO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        text += "감지되지 않음"
        cv2.putText(result_image, "NO SIGNAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print(text)
    
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
    
    # 노란색 원본 영역 시각화 (노란색 영역에 해당하는 원본 이미지를 추출)
    yellow_original = np.zeros_like(image)
    np.copyto(yellow_original[0:roi_height, 0:width], roi, where=mask_yellow[:,:,np.newaxis]>0)
    
    return result_image, mask_viz, mask_red_viz, mask_yellow_viz, mask_green_viz, roi_display, yellow_original

def save_debug_images(result_image, roi_display, mask_viz, mask_green_viz, mask_yellow_viz, yellow_original):
    """디버깅을 위해 이미지를 파일로 저장"""
    try:
        cv2.imwrite("/tmp/result_image.png", result_image)
        cv2.imwrite("/tmp/roi_display.png", roi_display)
        cv2.imwrite("/tmp/mask_viz.png", mask_viz)
        cv2.imwrite("/tmp/mask_green_viz.png", mask_green_viz)
        cv2.imwrite("/tmp/mask_yellow_viz.png", mask_yellow_viz)
        cv2.imwrite("/tmp/yellow_original.png", yellow_original)
        print("디버깅 이미지가 /tmp/ 폴더에 저장되었습니다.")
    except Exception as e:
        print(f"이미지 저장 오류: {e}")

# OpenCV 창 설정 시도
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

rospy.init_node('cam_test', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

rospy.wait_for_message("/usb_cam/image_raw/", Image)
print("카메라 준비됨 --------------")
print(f"OpenCV 버전: {cv2.__version__}")

# HSV 색상 범위 조정을 위한 트랙바 함수
def nothing(x):
    pass

# 이미지 표시 문제를 디버깅하기 위한 변수
frame_count = 0
save_debug_frame = 10  # 10번째 프레임에서 디버그 이미지 저장

while not rospy.is_shutdown():
    if cv_image.size != 0:
        try:
            frame_count += 1
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 신호등 색상 감지
            results = detect_traffic_light(cv_image)
            if results is None:
                print("유효하지 않은 이미지 처리 결과")
                continue
                
            result_image, mask_viz, mask_red_viz, mask_yellow_viz, mask_green_viz, roi_display, yellow_original = results
            
            # 주기적으로 디버그 이미지 저장 (매 100 프레임)
            if frame_count % 100 == 0:
                save_debug_images(result_image, roi_display, mask_viz, mask_green_viz, mask_yellow_viz, yellow_original)
            
            # 결과 표시 시도
            try:
                cv2.imshow("원본", cv_image)
                cv2.imshow("ROI 영역", roi_display)
                cv2.imshow("신호등 감지", result_image)
                cv2.imshow("색상 마스크", mask_viz)
                cv2.imshow("노란색 마스크", mask_yellow_viz)
                cv2.imshow("노란색 원본", yellow_original)
                
                # 표시 후 확인
                print(f"프레임 {frame_count}: 이미지 표시 시도") if frame_count % 100 == 0 else None
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC 키를 누르면 종료
                    break
            except Exception as e:
                print(f"이미지 표시 오류: {e}")
                # 오류 발생 시 디버그 이미지 저장
                save_debug_images(result_image, roi_display, mask_viz, mask_green_viz, mask_yellow_viz, yellow_original)
                
        except Exception as e:
            print(f"이미지 처리 오류: {e}")

# 종료 시 창 정리
cv2.destroyAllWindows()