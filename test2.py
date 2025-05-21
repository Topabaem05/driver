#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray # 차선 시각화를 위해 사용 가능
# from <your_custom_msgs.msg> import LaneOutput # 사용자 정의 메시지를 사용할 경우

# --- UFLD 모델 관련 설정 및 로드 ---
# 실제 UFLD 모델 파일 경로와 모델 정의 코드를 사용해야 합니다.
# 예시: from ultra_fast_lane_detector import UltraFastLaneDetector, ModelType

MODEL_PATH = "/path/to/your/ufld_model.pth" # 실제 모델 경로로 수정
MODEL_TYPE = "CULane" # 또는 "TuSimple", 모델에 맞게 수정
USE_GPU = torch.cuda.is_available()

class UFLDROSNode:
    def __init__(self):
        rospy.init_node('ufld_ros_node', anonymous=True)

        # UFLD 모델 로드 (이 부분은 실제 UFLD 구현에 따라 크게 달라집니다)
        # self.model = UltraFastLaneDetector(MODEL_PATH, model_type=ModelType[MODEL_TYPE], use_gpu=USE_GPU)
        # 아래는 단순 예시이며, 실제 모델 로딩 코드로 대체해야 합니다.
        self.model = self.load_ufld_model(MODEL_PATH)
        if self.model is None:
            rospy.logerr("Failed to load UFLD model.")
            return
        if USE_GPU:
            self.model.cuda()
        self.model.eval()
        rospy.loginfo("UFLD model loaded successfully.")

        self.bridge = CvBridge()

        # 구독자: 카메라 이미지
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("Subscribed to /camera/image_raw")

        # 발행자: 차선 검출 결과 이미지 (디버깅용)
        self.lane_image_pub = rospy.Publisher("/ufld/lane_image", Image, queue_size=1)
        rospy.loginfo("Publishing lane image to /ufld/lane_image")

        # 발행자: 차선 정보 (MarkerArray 또는 사용자 정의 메시지)
        self.lane_marker_pub = rospy.Publisher("/ufld/lane_markers", MarkerArray, queue_size=1)
        rospy.loginfo("Publishing lane markers to /ufld/lane_markers")

        # UFLD 모델 입력 이미지 크기 (모델에 따라 다름)
        self.input_width = 800 # 예시 값, 실제 모델에 맞게 수정
        self.input_height = 288 # 예시 값, 실제 모델에 맞게 수정


    def load_ufld_model(self, model_path):
        """
        실제 UFLD 모델 로딩 로직 구현.
        PyTorch 모델을 로드하고 반환합니다.
        UFLD 프로젝트의 모델 로딩 부분을 참고하세요.
        """
        try:
            # 예시: torch.load, 모델 아키텍처 정의 등
            # model = ...
            # model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
            # return model
            rospy.logwarn("`load_ufld_model` function needs to be implemented based on your UFLD model structure.")
            # 임시로 None을 반환하여 실행은 되도록 하지만, 실제로는 모델 객체를 반환해야 합니다.
            # 실제 사용 시 이 부분을 반드시 구현해야 합니다.
            # 아래는 GitHub에서 'ultrafastLaneDetector' PyTorch 구현체의 일반적인 로딩 방식 예시입니다.
            # from tusimple_transforms import Compose, ToTensor, Normalize, Resize
            # from ultrafastLaneDetector.model.model_culane import parsingNet # 또는 다른 모델
            #
            # 실제 모델 아키텍처에 맞는 클래스를 가져와야 합니다.
            # model = parsingNet(pretrained=False, backbone='18', cls_dim=(201, 18, 4), use_aux=False) # CULane 예시
            # if USE_GPU:
            #     state_dict = torch.load(model_path, map_location='cuda')['model']
            # else:
            #     state_dict = torch.load(model_path, map_location='cpu')['model']
            # model.load_state_dict(state_dict, strict=False) # strict=False가 필요할 수 있음
            # return model
            return None # <<-- 실제 모델 로딩 코드로 대체하세요!
        except Exception as e:
            rospy.logerr(f"Error loading UFLD model: {e}")
            return None

    def preprocess_image(self, cv_image):
        """
        UFLD 모델 입력에 맞게 이미지 전처리.
        (리사이즈, 정규화 등)
        """
        # 예시: UFLD의 일반적인 전처리 단계
        img = cv2.resize(cv_image, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        # 필요한 경우 추가적인 정규화 (예: ImageNet 통계 사용)
        # img = (img - mean) / std
        img = img.transpose(2, 0, 1) # HWC to CHW
        return torch.from_numpy(img).unsqueeze(0) # Add batch dimension

    def draw_lanes(self, image, lanes, color=(0,255,0), thickness=2):
        """
        검출된 차선을 원본 이미지에 그리는 함수 (단순 예시).
        UFLD 모델의 출력 형식에 따라 `lanes`의 구조가 다릅니다.
        UFLD는 보통 각 차선에 대해 x좌표들을 특정 y좌표에서 예측합니다.
        """
        if lanes is None or not lanes: # lanes가 None이거나 비어있을 경우
            return image

        img_h, img_w, _ = image.shape
        # UFLD 출력 (상대 좌표)을 절대 좌표로 변환하는 로직 필요
        # lanes는 보통 [[(x1,y1), (x2,y2), ...], [...]] 형태의 점들의 리스트일 수 있음
        # 또는 모델 출력 형식에 따라 다름 (예: 확률 맵에서 좌표 추출)
        # 아래는 UFLD가 이미지 높이에 따른 x좌표들의 리스트를 반환한다고 가정
        # 실제 UFLD 출력 형식에 맞춰 파싱해야 합니다.
        # 예시: lanes = [[x_coords_lane1], [x_coords_lane2], ...]
        # 여기서 y_coords는 보통 미리 정의된 샘플링 y 값들입니다.
        # (UFLD의 get_lanes 함수 참고)

        # 이 부분은 UFLD 모델의 `get_lanes` 또는 유사한 후처리 함수를 참고하여
        # 모델 출력(일반적으로 텐서)에서 실제 차선 좌표를 추출하는 로직으로 대체해야 합니다.
        # 아래는 단순 시각화를 위한 가정일 뿐, 실제 UFLD 모델 출력에 맞게 수정해야 합니다.
        # for lane_idx, lane_points_x in enumerate(lanes):
        #     # y_samples는 UFLD 모델이 사용하는 y축 샘플링 위치입니다.
        #     # y_samples = np.linspace(img_h * 0.4, img_h -1 , num_points_per_lane) # 예시
        #     for i in range(len(lane_points_x) - 1):
        #         # x1 = int(lane_points_x[i] * img_w) # UFLD 출력이 상대 좌표일 경우
        #         # y1 = int(y_samples[i])
        #         # x2 = int(lane_points_x[i+1] * img_w)
        #         # y2 = int(y_samples[i+1])
        #         # if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        #         #     cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        pass # <<-- 실제 차선 그리기 로직으로 대체하세요!
        rospy.logwarn_once("`draw_lanes` function needs to be implemented based on your UFLD model output format.")
        return image


    def create_lane_markers(self, lanes, header):
        """
        검출된 차선을 visualization_msgs/MarkerArray로 변환.
        UFLD 출력 형식에 맞게 수정 필요.
        """
        marker_array = MarkerArray()
        if lanes is None:
            return marker_array

        # 이 부분도 UFLD 모델의 출력 형식에 따라 `lanes`를 파싱해야 합니다.
        # 아래는 차선이 포인트 리스트로 주어진다고 가정한 예시입니다.
        # for lane_idx, lane_points in enumerate(lanes): # lanes = [[(x1,y1,z1), (x2,y2,z2)], ...]
        #     if not lane_points:
        #         continue
        #
        #     marker = Marker()
        #     marker.header = header
        #     marker.ns = "lanes"
        #     marker.id = lane_idx
        #     marker.type = Marker.LINE_STRIP
        #     marker.action = Marker.ADD
        #     marker.scale.x = 0.1 # 선 두께
        #     marker.color.a = 1.0 # 불투명도
        #     marker.color.r = 0.0
        #     marker.color.g = 1.0
        #     marker.color.b = 0.0
        #
        #     for pt_3d in lane_points: # pt_3d = (x,y,z) 카메라 좌표계 기준
        #         p = Point()
        #         p.x = pt_3d[0]
        #         p.y = pt_3d[1]
        #         p.z = pt_3d[2] # 2D 이미지에서는 z=0 또는 다른 값으로 설정
        #         marker.points.append(p)
        #
        #     marker_array.markers.append(marker)
        rospy.logwarn_once("`create_lane_markers` function needs to be implemented based on your UFLD model output format and desired 3D representation.")
        return marker_array


    def image_callback(self, ros_image):
        if self.model is None:
            return

        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image_original = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 이미지 전처리
        input_tensor = self.preprocess_image(cv_image_original.copy())
        if USE_GPU:
            input_tensor = input_tensor.cuda()

        # UFLD 모델 추론
        with torch.no_grad():
            # 이 부분은 UFLD 모델의 forward pass입니다.
            # model_output = self.model(input_tensor)
            # 실제 UFLD 모델은 보통 딕셔너리 형태로 출력을 반환하거나, 텐서를 직접 반환합니다.
            # 예: out = self.model(input_tensor) # out은 (batch_size, num_lanes * num_points_y + num_classes, grid_cells_y, grid_cells_x)
            # 또는 out, aux_seg_out = self.model(input_tensor)
            # 여기서는 UFLD의 후처리 (예: `get_lanes` 함수)를 통해
            # 실제 차선 좌표를 얻는 과정이 필요합니다.
            # `detected_lanes`는 모델의 출력을 파싱한 결과여야 합니다.
            # 예: detected_lanes = self.model.get_lanes(model_output, as_json=False)
            # 실제 UFLD 구현체의 추론 및 후처리 코드를 여기에 적용해야 합니다.
            rospy.logwarn_once("UFLD model inference and post-processing logic needs to be implemented.")
            model_output_placeholder = None # <<-- 실제 모델 추론 결과로 대체
            detected_lanes = None # <<-- 실제 후처리된 차선 정보로 대체

        # 결과 시각화 (디버깅용)
        debug_image = cv_image_original.copy()
        # `detected_lanes`의 형식은 UFLD의 `get_lanes` 함수 출력을 따릅니다.
        # (일반적으로 이미지 높이별 x좌표 리스트의 리스트)
        self.draw_lanes(debug_image, detected_lanes)

        try:
            self.lane_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # 차선 정보를 MarkerArray로 변환하여 발행
        # `detected_lanes`를 3D 포인트로 변환하는 로직이 필요할 수 있습니다.
        # (카메라 외부 파라미터 등을 이용한 역투영 변환 등)
        # 여기서는 단순화를 위해 2D 좌표를 그대로 사용하거나, z=0으로 가정합니다.
        # lane_markers = self.create_lane_markers(detected_lanes_for_markers, ros_image.header)
        # self.lane_marker_pub.publish(lane_markers)


if __name__ == '__main__':
    try:
        node = UFLDROSNode()
        if node.model is not None: # 모델 로딩 성공 시에만 spin
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in main: {e}")