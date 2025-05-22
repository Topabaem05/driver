# Advanced Lane Detection and Control for Xycar

This project implements a ROS-based autonomous driving system for the Xycar platform, featuring an advanced lane detection pipeline and a PID with Feed-Forward steering controller.

## Core Functionality

The system processes camera images to detect lane lines, calculate lane geometry, and control the vehicle's steering and speed. The main pipeline stages are:

1.  **Image Preprocessing (`preprocess_image`):**
    *   Selects a Region of Interest (ROI) from the input camera image.
    *   Converts the ROI to HLS and HSV color spaces.
    *   Extracts individual H, L, S channels from HLS.

2.  **Mask Creation (`create_masks`):**
    *   Generates binary masks for white and yellow lane lines using HSV color thresholding.
    *   Creates masks from HLS L-channel (brightness) and S-channel (saturation).
    *   Applies Sobel edge detection (x-derivative on L-channel) and creates an edge mask.
    *   Combines these masks (HSV color OR (S-channel AND Sobel edge)) to create a robust binary image highlighting potential lane features.

3.  **Perspective Transformation (`perspective_transform`):**
    *   Warps the combined binary mask into a bird's-eye view using a perspective transform.
    *   Calculates both the forward (`M`) and inverse (`Minv`) transformation matrices.

4.  **Lane Pixel Localization:**
    *   **Histogram Analysis (`find_lane_starting_points`):** Calculates a histogram of the bottom half of the warped image to find initial base x-positions for the left and right lane lines.
    *   **Sliding Window Search (`sliding_window_search`):** If sufficient pixels are found by the histogram, this method uses a series of sliding windows, starting from the identified base positions, to trace and collect non-zero pixels belonging to each lane line up the height of the image. Windows are re-centered based on the mean position of detected pixels.

5.  **Polynomial Fitting (`fit_polynomial`):**
    *   Fits 2nd order polynomials (ax^2 + bx + c) to the detected left and right lane pixel coordinates (from sliding window or previous frame).
    *   Includes logic to use lane fits from the previous frame if the current frame's detection is weak (too few pixels).
    *   Calculates the radius of curvature for each lane line and an average curvature (in meters).
    *   Determines the vehicle's lateral offset (`center_diff`) from the center of the detected lane (in meters).

6.  **Heading Error Calculation (`calculate_heading_error`):**
    *   Calculates the lane's center line from the fitted left and right polynomials.
    *   Fits a new polynomial to this center line in real-world meter space.
    *   Determines the tangent to this center line at the bottom of the warped image.
    *   Calculates the heading error (`e_psi`) as the arctangent of this tangent, representing the angle between the vehicle's path and the lane's direction.
    *   Also computes a `signed_curvature_factor` (Am coefficient from the center line's meter-space polynomial) used for feed-forward control.

7.  **Steering Control (PID + Feed-Forward in `process_image`):**
    *   Calculates the derivative of the vehicle offset (`d_offset_dt`).
    *   **PID Terms for Lateral Control:**
        *   Proportional term based on `center_diff` (vehicle's lateral offset from lane center).
        *   Integral term based on the accumulated `center_diff` to correct steady-state errors. (Includes anti-windup).
        *   Derivative term based on the rate of change of `center_diff` (`d_offset_dt`) to dampen response and improve stability.
        *   Proportional term based on `heading_error_rad` (angle between vehicle's path and lane's tangent).
    *   **Feed-Forward Term:**
        *   Based on the `signed_curvature_factor` (Am coefficient from the centerline's meter-space polynomial) to proactively steer into curves.
    *   The final steering angle is a sum of these terms, converted to degrees, then clipped (e.g., to +/-35 degrees) and smoothed.

8.  **Speed Control (Curvature-Aware in `process_image`):**
    *   Determines a `base_speed_for_curve` based on the average lane curvature radius (slower for sharper curves).
    *   Calculates a `speed_due_to_angle` based on the magnitude of the final steering angle (slower for larger steering angles).
    *   The final speed is the minimum of these two values.

9.  **Visualization (`visualize_lanes`):**
    *   Draws the detected lane area (filled polygon) and fitted lane lines (as dots) on a bird's-eye view image.
    *   Unwarps this visualization back to the original camera perspective.
    *   Overlays the unwarped lane graphics onto the original camera image.
    *   Displays text information: average curvature radius, vehicle offset, and final steering angle.
    *   Additional debug windows show intermediate images (e.g., combined mask, warped image).

## Prerequisites

### Software
- **ROS (Robot Operating System):** e.g., Melodic, Noetic.
- **Python 3**
- **OpenCV (`cv2`)**
- **NumPy**
- **ROS Packages:** `cv_bridge`, `sensor_msgs`, `xycar_msgs` (for `XycarMotor.msg`).
- **Matplotlib (Optional for Debugging):** Used for plotting histograms if available.

### Hardware
- **Xycar Platform** (or similar ROS-compatible robot).
- **Camera.**

## Installation & Setup

1.  **ROS Workspace:**
    - Clone or place this package (e.g., `xycar_advanced_control`) into your Catkin workspace's `src/` directory.
2.  **Python Dependencies:**
    ```bash
    pip install numpy opencv-python matplotlib
    ```
3.  **ROS Dependencies:**
    - Install `cv_bridge` (e.g., `sudo apt-get install ros-<distro>-cv-bridge`).
    - Ensure `xycar_msgs` is built in your workspace.
4.  **Build Workspace:**
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## How to Run

1.  **Start ROS Core:** `roscore`
2.  **Launch Camera Node.**
3.  **Run the Lane Controller Node:**
    ```bash
    # Example, replace 'xycar_advanced_control' with your package name
    rosrun xycar_advanced_control sam_lane_keeping_xycar.py 
    ```
    *Note: Consider renaming `sam_lane_keeping_xycar.py` to a name more reflective of its current advanced capabilities, like `advanced_lane_controller.py`.*

### ROS Topics
- **Subscribed:** `/usb_cam/image_raw` (`sensor_msgs/Image`)
- **Published:** `/xycar_motor` (`xycar_msgs/XycarMotor`)

### Key Parameters & Tuning

The following gains for the PID and Feed-Forward controller are defined at the top of `sam_lane_keeping_xycar.py` and **require careful tuning** for optimal performance in your specific environment:

- `KP_OFFSET`: Proportional gain for lateral offset. Controls how strongly the car reacts to being off-center.
- `KI_OFFSET`: Integral gain for lateral offset. Helps eliminate steady-state errors (e.g., consistent drift to one side). (Includes anti-windup).
- `KD_OFFSET`: Derivative gain for lateral offset. Dampens oscillations and smooths the response to changes in offset.
- `KP_HEADING`: Proportional gain for heading error. Controls how strongly the car corrects its angle relative to the lane.
- `K_FF_CURVATURE`: Feed-forward gain for lane curvature. Allows the car to anticipate turns based on detected road curvature.

Tuning these values is an iterative process. It's often recommended to tune P gains first, then D, then I, and finally the FF gain. Other parameters like ROI ratios, color thresholds, and perspective transform points (`SRC_POINTS_ROI_RATIOS`) also significantly impact performance and may need adjustment.
