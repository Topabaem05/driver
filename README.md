# Xycar Lane Keeping Assistant

This project implements a ROS-based lane-keeping assistance system for the Xycar platform. It uses computer vision techniques to detect lane lines and control the vehicle's steering to stay within its lane.

## Core Functionality

- **Image Processing:**
  - Selects a Region of Interest (ROI) from the input camera image to focus on the road ahead.
  - Converts the ROI to the HSV color space for more robust line detection under varying lighting conditions.
- **Lane Line Detection:**
  - Detects white and yellow lane lines using color thresholding in the HSV space.
  - Generates binary masks for white and yellow lines.
- **Steering Control:**
  - **White Line Tracking:** Calculates an error based on the ratio of detected white pixels on the left and right sides of the ROI. It also uses the center of the detected white track to refine the steering angle.
  - **Yellow Line Fallback:** If white lines are not sufficiently detected, the system attempts to track a yellow line, calculating steering based on its position.
  - **"No Line" State:** If no lines are detected, the vehicle may be programmed to drive straight or hold the last known steering angle (currently drives straight).
- **Speed Control:**
  - Implements a basic dynamic speed adjustment strategy:
    - Higher speeds when driving straight or with small steering angles.
    - Reduced speeds for sharper turns, when line detection is poor ("OUTSIDE WARNING"), or when no lines are found.
- **Steering Smoothing:**
  - Limits the maximum change in steering angle per frame to ensure smoother vehicle motion.
- **ROS Integration:**
  - Operates as a ROS node.
  - Subscribes to a camera image topic (e.g., `/usb_cam/image_raw`).
  - Publishes motor commands (angle and speed) to `/xycar_motor` using `xycar_msgs/XycarMotor`.
- **Visual Feedback:**
  - Displays OpenCV windows showing:
    - The original camera image.
    - The selected Region of Interest (ROI).
    - The binary masks for white and yellow lines.
  - Logs status information (current mode, angle, speed) to the console via `rospy.loginfo`.

## Prerequisites

### Software

- **Robot Operating System (ROS):**
  - E.g., ROS Melodic (Ubuntu 18.04) or ROS Noetic (Ubuntu 20.04).
  - Core ROS packages: `rospy`, `std_msgs`, `sensor_msgs`.
- **Python:**
  - Python 3.6+
- **Python Libraries:**
  - **OpenCV (`cv2`):** `pip install opencv-python`
  - **NumPy:** `pip install numpy`
- **ROS Packages:**
  - **`cv_bridge`:** For ROS image conversion. Install via `sudo apt-get install ros-<your_ros_distro>-cv-bridge`.
  - **`xycar_msgs`:** Custom messages for Xycar control (specifically `XycarMotor.msg`). Typically provided with Xycar setup.

### Hardware

- **Xycar Platform:** Or a similar ROS-compatible differential drive robot.
- **Camera:** A monocular camera providing a raw image feed.

## Installation & Setup

1.  **ROS Workspace:**
    - Create or use an existing Catkin workspace (e.g., `~/catkin_ws`).
    - Place this package (e.g., named `xycar_lane_assist`) into `~/catkin_ws/src/`.
      ```bash
      cd ~/catkin_ws/src/
      # git clone <repository_url> xycar_lane_assist # Or copy package here
      ```
2.  **Python Dependencies:**
    ```bash
    pip install numpy opencv-python
    ```
3.  **ROS Dependencies:**
    - **`cv_bridge`:**
      ```bash
      sudo apt-get update
      sudo apt-get install ros-<your_ros_distro>-cv-bridge
      ```
    - **`xycar_msgs`:** Ensure this package is in your workspace and built if it's not globally installed.
4.  **Build Workspace:**
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## How to Run

1.  **Start ROS Core:**
    ```bash
    roscore
    ```
2.  **Launch Camera Node (if applicable):**
    - Ensure your camera is publishing to the `/usb_cam/image_raw` topic (or as configured in the script).
    - Example: `roslaunch usb_cam usb_cam-test.launch`
3.  **Run the Lane Keeping Node:**
    - In a new terminal (after sourcing workspace):
      ```bash
      rosrun xycar_lane_assist sam_lane_keeping_xycar.py 
      # Replace 'xycar_lane_assist' with your actual package name.
      # The script itself is now 'sam_lane_keeping_xycar.py' but contains the lane keeping logic.
      # Consider renaming the .py file to something like 'lane_keeping_node.py' for clarity.
      ```

### ROS Topics

- **Subscribed:** `/usb_cam/image_raw` (`sensor_msgs/Image`)
- **Published:** `/xycar_motor` (`xycar_msgs/XycarMotor`)
