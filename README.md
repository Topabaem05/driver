# SAM-based Collision Avoidance and Lane Keeping for Xycar

This project implements a ROS-based autonomous driving system for the Xycar platform. It features:
- Real-time rubber cone detection using a (currently mocked) Segment Anything Model (SAM).
- Collision avoidance maneuvers based on the Collision Cone methodology.
- Lane keeping assistance by detecting yellow centerlines and white solid lines.

The system is designed to perceive its environment, make decisions to avoid obstacles, and maintain its lane, all within a ROS framework.

## Core Components & Features

- **Rubber Cone Segmentation (Mocked SAM):**
  - Utilizes a simulated Segment Anything Model (SAM) for zero-shot detection of rubber cones.
  - Automatic Mask Generation is conceptually used to identify all cones in an image.
  - Bounding boxes and centers of mass are calculated from masks for 2D cone localization.
- **Collision Cone Algorithm:**
  - Implements the Collision Cone method to determine if a detected cone poses an imminent collision threat.
  - Calculates relative positions, safety radii, and critical angles (`theta` and `alpha`).
- **Avoidance Maneuvering:**
  - If a collision is predicted (`alpha < theta`), the system calculates an avoidance steering angle (`phi`).
  - Sends steering commands via ROS to perform the maneuver.
- **Lane Keeping Assist:**
  - Detects yellow centerlines and white lane lines using HSV color space filtering and contour analysis.
  - Calculates steering adjustments to keep the vehicle centered in its lane.
- **ROS Integration:**
  - Operates as a ROS node, subscribing to camera image topics (e.g., `/usb_cam/image_raw`).
  - Publishes motor commands (angle and speed) to a Xycar topic (e.g., `/xycar_motor`).
- **Visual Debugging:**
  - Displays the original image with overlays:
    - Detected cones (bounding boxes and centers).
    - Current operational mode (avoidance, lane keeping status).
    - Final steering angle.
  - Shows processed white and yellow line masks.
- **Mocked Components for Development:**
  - SAM functionality and sensor data (vehicle pose, obstacle world positions) are currently mocked, allowing for development and testing of the core logic without requiring full hardware setup. Placeholder comments in the code guide future integration.

## Prerequisites & Requirements

### Software

- **Robot Operating System (ROS):**
  - Tested on ROS Melodic (Ubuntu 18.04) but should be compatible with ROS Noetic (Ubuntu 20.04).
  - Core ROS packages: `rospy`, `std_msgs`, `sensor_msgs`.
- **Python:**
  - Python 3.6+
- **Python Libraries:**
  - **OpenCV (`cv2`):** For image processing. Install via `pip install opencv-python`.
  - **NumPy:** For numerical operations. Install via `pip install numpy`.
- **ROS Packages:**
  - **`cv_bridge`:** For converting between ROS image messages and OpenCV images.
    - Install via `sudo apt-get install ros-<your_ros_distro>-cv-bridge` (e.g., `ros-melodic-cv-bridge`).
  - **`xycar_msgs`:** Custom ROS messages for Xycar control (typically `XycarMotor.msg`).
    - These are usually provided with Xycar-specific packages and built in your Catkin workspace.
  - **`image_transport`:** Often used with `cv_bridge` for publishing and subscribing to image topics. (Usually part of a standard ROS installation).

### Hardware (Conceptual for this Project)

- **Xycar Platform:** Or a similar differential drive robot compatible with ROS.
- **Camera:** A monocular camera providing a raw image feed (e.g., USB webcam).
- **(Future/Real Implementation) SAM-compatible Processing Unit:** A GPU or other hardware capable of running the Segment Anything Model efficiently if real SAM is integrated.
- **(Future/Real Implementation) Advanced Sensors:** For accurate real-world operation:
  - **Depth Sensing:** LiDAR or Stereo Camera for accurate distance measurement to obstacles.
  - **Localization:** IMU, GPS, Wheel Encoders for precise vehicle pose and velocity.

### ROS Message Types

- **Input:** `sensor_msgs/Image` (for the camera feed).
- **Output:** `xycar_msgs/XycarMotor` (for sending steering angle and speed to the Xycar).

## Installation & Setup

1.  **Set up ROS Workspace:**
    - Ensure you have a Catkin workspace (e.g., `~/catkin_ws`).
    - Clone or place this project package (let's assume its name is `sam_drive_assist`) into the `~/catkin_ws/src/` directory.
      ```bash
      cd ~/catkin_ws/src/
      # git clone <repository_url> sam_drive_assist  # If cloning
      # Or copy the package folder here
      ```

2.  **Install Python Dependencies:**
    - Navigate to the project directory or ensure your Python environment has the required libraries:
      ```bash
      pip install numpy opencv-python
      ```

3.  **Install ROS Dependencies:**
    - **`cv_bridge`:** If not already installed with your ROS distribution:
      ```bash
      sudo apt-get update
      sudo apt-get install ros-<your_ros_distro>-cv-bridge 
      # Replace <your_ros_distro> with your ROS version (e.g., melodic, noetic)
      ```
    - **`xycar_msgs`:** This package provides custom messages for Xycar control.
      - It's typically provided as part of the Xycar educational toolkit.
      - If you have the `xycar_msgs` source, place it in your `~/catkin_ws/src/` directory and build it.

4.  **Build the Workspace:**
    - Compile your Catkin workspace:
      ```bash
      cd ~/catkin_ws
      catkin_make
      ```
    - Source your workspace:
      ```bash
      source devel/setup.bash 
      # Or add this to your .bashrc
      ```

5.  **Segment Anything Model (SAM) - For Future Real Implementation:**
    - The current script uses a **mocked** SAM.
    - For a real SAM integration:
      - You would need to install SAM according to its official repository (e.g., [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)).
      - Download the appropriate pre-trained model weights.
      - Modify the `detect_cones_sam` function in the script to call the actual SAM.
      - This would likely require a machine with a compatible GPU for reasonable performance.

## How to Run the Node

1.  **Start ROS Core:**
    - Open a new terminal and run:
      ```bash
      roscore
      ```

2.  **Launch the Driving Node:**
    - Open another terminal.
    - Source your Catkin workspace (if you haven't added it to your `.bashrc`):
      ```bash
      source ~/catkin_ws/devel/setup.bash
      ```
    - Run the Python script using `rosrun`:
      ```bash
      rosrun sam_drive_assist sam_lane_keeping_xycar.py
      ```
      *(Replace `sam_drive_assist` with your actual package name if different.)*

3.  **Ensure Camera Feed (if not simulated):**
    - If you are using a real camera, make sure it is connected and publishing images to the `/usb_cam/image_raw` topic (or the topic configured in the script).
    - You might need to launch your camera node, e.g.:
      ```bash
      roslaunch usb_cam usb_cam-test.launch
      ```
      *(This command can vary based on your camera setup.)*

4.  **Monitor Output:**
    - The script will print status information (Mode, Angle, Speed) to the terminal.
    - OpenCV windows will display:
      - The original camera feed with detected cones and status text.
      - The white line mask.
      - The yellow line mask.

### ROS Topics

- **Subscribed Topics:**
  - `/usb_cam/image_raw` (Type: `sensor_msgs/Image`): Input camera image.
- **Published Topics:**
  - `/xycar_motor` (Type: `xycar_msgs/XycarMotor`): Steering angle and speed commands.

## Mocked Components & Future Work

It is crucial to understand that key parts of this system are currently **mocked** for development and testing purposes. To deploy this in a real-world scenario, these components would need to be replaced with actual implementations.

- **Segment Anything Model (SAM) for Cone Detection:**
  - The `detect_cones_sam` function in `sam_lane_keeping_xycar.py` (originating from `collision_avoidance_helpers.py`) **does not** perform real SAM inference.
  - It returns predefined, hardcoded cone data (bounding boxes, centers of mass).
  - **Future Work:** Integrate a real SAM model. This involves loading the model, performing inference on camera images, and processing the output masks to extract cone information. Refer to the `TODO` comments in the code.

- **Sensor Data (Vehicle Pose & Obstacle Positions):**
  - The `get_cone_distance_and_vehicle_pose` function also returns **mocked data**.
  - It provides fixed values for the vehicle's position (`pos_vehicle`), velocity (`vel_vehicle`), and the world position of a detected obstacle (`pos_obstacle_world`).
  - **Future Work:**
    - Integrate real sensors (IMU, GPS, wheel encoders) for accurate vehicle localization and velocity.
    - For `pos_obstacle_world`:
      - Use depth sensors (LiDAR, stereo camera, or monocular depth estimation) to find the distance to each cone detected by SAM.
      - Implement coordinate transformations to convert the 2D pixel coordinates of cones (from SAM) plus their depth into 3D world coordinates relative to the vehicle or a global map. Refer to the `TODO` comments in the code.

The existing logic for collision avoidance and lane keeping is built upon these mocked components. The primary focus of this project version is the decision-making and control logic, assuming perception data is available.
