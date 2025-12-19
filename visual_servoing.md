# Laboratory: Visual Servoing for Mobile Robots

## Learning Objectives

By the end of this laboratory, students will be able to:

1. Understand the fundamental principles of visual servoing and its applications in robotics
2. Distinguish between Image-Based Visual Servoing (IBVS) and Position-Based Visual Servoing (PBVS)
3. Implement basic visual servoing controllers for mobile robot navigation
4. Analyze the performance characteristics and trade-offs of different visual servoing approaches
5. Design and tune control laws for vision-guided robot motion

## Prerequisites

- ROS2 (Jazzy or Humble)
- Python 3.10+
- OpenCV (cv2)
- NumPy
- Basic understanding of camera geometry and homogeneous transformations
- Familiarity with PID control

## Duration

3 hours

---

## Part 1: Theoretical Background

### 1.1 Introduction to Visual Servoing

Visual servoing is a control technique that uses visual feedback from cameras to control the motion of a robot. Unlike traditional position-based control that relies on encoder feedback or external positioning systems, visual servoing directly uses image features to compute control commands.

**Key advantages:**

- Direct use of sensor information without intermediate pose estimation
- Robustness to calibration errors in some configurations
- Natural task specification in image space

**Applications:**

- Object manipulation and grasping
- Autonomous navigation
- Surveillance and tracking
- Precision assembly

### 1.2 Image-Based Visual Servoing (IBVS)

In IBVS, the control law is computed directly in the image space. The error is defined as the difference between current image features and desired image features.

**Advantages:**

- No 3D reconstruction required
- Potentially more accurate in image space
- Direct control of visual features

**Disadvantages:**

- Requires depth estimation or approximation
- Camera retreat problem (robot may move away from target)
- Local minima in presence of large errors

### 1.3 Position-Based Visual Servoing (PBVS)

In PBVS, visual information is used to estimate the 3D pose of the target relative to the camera. The control law is then computed in Cartesian space.

**Control formulation:**

```
v = -λ (r - r*)
ω = -λ θu
```

Where:

- `v` = linear velocity
- `ω` = angular velocity
- `r` = current position
- `r*` = desired position
- `θu` = rotation error (angle-axis representation)

**Advantages:**

- Intuitive behavior in 3D space
- Easier to predict robot trajectories
- No camera retreat problem
- Decoupled translation and rotation control

**Disadvantages:**

- Requires accurate 3D pose estimation
- Sensitive to calibration errors
- Feature points may leave field of view during motion

### 1.4 Comparison: IBVS vs PBVS

| Aspect                  | IBVS                    | PBVS                 |
| ----------------------- | ----------------------- | -------------------- |
| Control space           | Image plane (2D)        | Cartesian space (3D) |
| Pose estimation         | Not required            | Required             |
| Trajectory              | Complex, image-centered | Straight line in 3D  |
| Calibration sensitivity | Lower                   | Higher               |
| Field of view           | Better maintained       | May lose features    |
| Computational cost      | Lower                   | Higher               |
| Convergence domain      | Smaller (local)         | Larger (global)      |

### 1.5 Controller Design

Both IBVS and PBVS require appropriate controller design. Common approaches include:

**Proportional Control:**

```
u = -Kp * e
```

Simple but may have steady-state error.

**Proportional-Derivative Control:**

```
u = -Kp * e - Kd * ė
```

Adds damping, improves stability.

**Adaptive Control:**
Adjusts gains based on system performance or estimated parameters (e.g., depth in IBVS).

**Considerations:**

- Stability analysis using Lyapunov theory
- Convergence rate vs. control effort trade-off
- Robustness to noise and modeling errors

---

## Part 2: Laboratory Setup

### 2.1 Workspace Preparation

Create a ROS2 workspace for this laboratory:

```bash
mkdir -p ~/ros2_ws/src/visual_servoing_lab
cd ~/ros2_ws/src/visual_servoing_lab
```

### 2.2 Required Packages

Install dependencies:

```bash
sudo apt update
sudo apt install ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-geometry-msgs
pip3 install numpy scipy
```

### 2.3 Hardware Setup

**Required:**

- Mobile robot with differential drive
- Camera mounted on robot (ArUco TF publisher already running)
- 2D LiDAR sensor
- ArUco markers placed in environment

**Coordinate Frames:**

- `base_link`: Robot base frame
- `camera_link`: Camera frame
- `laser`: LiDAR frame
- `aruco_marker_X`: Marker frames (published by your TF publisher)

### 2.4 Verify System

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Verify marker detection
ros2 run tf2_ros tf2_echo camera_link aruco_marker_0

# Verify LiDAR data
ros2 topic echo /scan --once
```

---

## Part 3: Demo - Complete Implementation

This section provides complete, working implementations of both IBVS and PBVS controllers that you can run immediately.

### 3.1 Base Class: TF Listener

Create `marker_tf_listener.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from scipy.spatial.transform import Rotation

class MarkerTFListener(Node):
    """Base class for listening to ArUco marker transforms"""

    def __init__(self, node_name):
        super().__init__(node_name)

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.declare_parameter('target_marker_id', 0)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('base_frame', 'base_link')

        self.target_marker_id = self.get_parameter('target_marker_id').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.target_frame = f'aruco_marker_{self.target_marker_id}'

        self.get_logger().info(f'Initialized - Tracking marker {self.target_marker_id}')

    def get_marker_transform(self):
        """Get current transform from camera to marker"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            return transform
        except TransformException as ex:
            return None

    def transform_to_pose(self, transform):
        """Extract position and orientation from transform"""
        if transform is None:
            return None, None

        # Position
        position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        # Orientation as quaternion
        quat = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])

        return position, quat

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        rotation = Rotation.from_quat(quat)
        return rotation.as_euler('xyz')
```

### 3.2 Demo Controller: Position-Based Visual Servoing

Create `demo_pbvs_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
import numpy as np
from marker_tf_listener import MarkerTFListener

class DemoPBVSController(MarkerTFListener):
    """
    PROFESSOR DEMO: Position-Based Visual Servoing

    This controller demonstrates:
    - Reading marker pose from TF
    - Computing 3D position error
    - Computing orientation error (yaw)
    - Proportional control for both linear and angular velocities
    """

    def __init__(self):
        super().__init__('demo_pbvs_controller')

        # Control gains
        self.declare_parameter('kp_linear', 0.4)
        self.declare_parameter('kp_angular', 0.8)

        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value

        # Desired pose relative to marker (camera frame)
        self.declare_parameter('desired_distance', 0.5)  # 50cm from marker
        self.declare_parameter('desired_lateral_offset', 0.0)  # centered

        self.desired_z = self.get_parameter('desired_distance').value
        self.desired_x = self.get_parameter('desired_lateral_offset').value
        self.desired_y = 0.0

        # Goal tolerances
        self.position_tolerance = 0.05  # 5cm
        self.yaw_tolerance = 0.1  # ~5.7 degrees

        # Velocity limits
        self.max_linear_vel = 0.3  # m/s
        self.max_angular_vel = 1.0  # rad/s

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('=== PBVS DEMO Controller Ready ===')
        self.get_logger().info(f'Target: z={self.desired_z}m, x={self.desired_x}m')
        self.get_logger().info(f'Gains: kp_linear={self.kp_linear}, kp_angular={self.kp_angular}')

    def control_loop(self):
        """Main control loop"""

        # Get marker transform
        transform = self.get_marker_transform()

        if transform is None:
            # Marker not visible - stop robot
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().warn('Marker not visible!', throttle_duration_sec=2.0)
            return

        # Extract position and orientation
        position, quat = self.transform_to_pose(transform)

        # Current position in camera frame
        x_current = position[0]
        y_current = position[1]
        z_current = position[2]

        # Compute position errors
        error_x = self.desired_x - x_current
        error_y = self.desired_y - y_current
        error_z = self.desired_z - z_current

        # Total position error
        position_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

        # Compute desired yaw (angle to face marker)
        # For mobile robot: we want to rotate to align with marker
        desired_yaw = -np.arctan2(y_current, z_current)

        # Current yaw from quaternion
        euler = self.quaternion_to_euler(quat)
        current_yaw = euler[2]

        # Yaw error (simplified for demo)
        yaw_error = desired_yaw

        # Normalize to [-pi, pi]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # Create control command
        cmd = Twist()

        # Check if goal reached
        if position_error < self.position_tolerance and abs(yaw_error) < self.yaw_tolerance:
            self.get_logger().info('✓ GOAL REACHED!', throttle_duration_sec=2.0)
            self.cmd_vel_pub.publish(cmd)
            return

        # Proportional control
        # Linear velocity: move forward/backward based on depth error
        cmd.linear.x = self.kp_linear * error_z

        # Angular velocity: rotate to face marker
        cmd.angular.z = self.kp_angular * yaw_error

        # Apply velocity limits
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_linear_vel, self.max_linear_vel)
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_vel, self.max_angular_vel)

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log status
        self.get_logger().info(
            f'PBVS | Pos: [{x_current:.2f}, {y_current:.2f}, {z_current:.2f}]m | '
            f'Err: pos={position_error:.3f}m, yaw={np.degrees(yaw_error):.1f}° | '
            f'Cmd: v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
            throttle_duration_sec=0.5
        )

def main(args=None):
    rclpy.init(args=args)
    controller = DemoPBVSController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        controller.cmd_vel_pub.publish(Twist())
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.3 Demo Controller: Image-Based Visual Servoing

Create `demo_ibvs_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Twist
import numpy as np
from marker_tf_listener import MarkerTFListener

class DemoIBVSController(MarkerTFListener):
    """
    PROFESSOR DEMO: Image-Based Visual Servoing

    This controller demonstrates:
    - Computing image features (pixel coordinates) from 3D pose
    - Defining desired features in image space
    - Proportional control in image space
    - Handling depth estimation
    """

    def __init__(self):
        super().__init__('demo_ibvs_controller')

        # Control gain
        self.declare_parameter('lambda_gain', 0.5)
        self.lambda_gain = self.get_parameter('lambda_gain').value

        # Desired image features (pixels)
        self.declare_parameter('desired_u', 320.0)  # center of 640px image
        self.declare_parameter('desired_v', 240.0)  # center of 480px image
        self.declare_parameter('desired_depth', 0.5)  # 50cm depth

        self.u_desired = self.get_parameter('desired_u').value
        self.v_desired = self.get_parameter('desired_v').value
        self.z_desired = self.get_parameter('desired_depth').value

        # Camera intrinsics
        self.camera_matrix = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Current depth estimate
        self.Z = self.z_desired  # Initial estimate

        # Tolerance
        self.pixel_tolerance = 20  # pixels
        self.depth_tolerance = 0.05  # meters

        # Velocity limits
        self.max_linear_vel = 0.3
        self.max_angular_vel = 1.0

        # Subscribers and publishers
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('=== IBVS DEMO Controller Ready ===')
        self.get_logger().info(f'Target: u={self.u_desired}px, v={self.v_desired}px, z={self.z_desired}m')

    def camera_info_callback(self, msg):
        """Get camera calibration parameters"""
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            self.camera_matrix = K
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.get_logger().info(
                f'Camera calibrated: fx={self.fx:.1f}, fy={self.fy:.1f}')

    def control_loop(self):
        """Main control loop"""

        # Wait for camera calibration
        if self.camera_matrix is None:
            self.get_logger().warn('Waiting for camera info...', throttle_duration_sec=2.0)
            return

        # Get marker transform
        transform = self.get_marker_transform()

        if transform is None:
            # Marker not visible
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().warn('Marker not visible!', throttle_duration_sec=2.0)
            return

        # Extract 3D position
        position, _ = self.transform_to_pose(transform)
        x, y, z = position

        # Update depth estimate (adaptive)
        self.Z = z

        # Project 3D point to image plane
        u_current = self.fx * (x / z) + self.cx
        v_current = self.fy * (y / z) + self.cy

        # Compute image space errors
        error_u = u_current - self.u_desired
        error_v = v_current - self.v_desired
        error_z = z - self.z_desired

        # Image space error magnitude
        image_error = np.sqrt(error_u**2 + error_v**2)

        # Check if goal reached
        if image_error < self.pixel_tolerance and abs(error_z) < self.depth_tolerance:
            self.get_logger().info('✓ GOAL REACHED!', throttle_duration_sec=2.0)
            self.cmd_vel_pub.publish(Twist())
            return

        # Compute normalized image coordinates
        x_norm = (u_current - self.cx) / self.fx

        # Control law (simplified for differential drive)
        cmd = Twist()

        # Angular velocity: center marker horizontally in image
        # Negative because positive image x corresponds to negative robot rotation
        cmd.angular.z = -self.lambda_gain * x_norm * 2.5

        # Linear velocity: approach to desired depth
        cmd.linear.x = -self.lambda_gain * error_z

        # Apply velocity limits
        cmd.linear.x = np.clip(cmd.linear.x, -self.max_linear_vel, self.max_linear_vel)
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_vel, self.max_angular_vel)

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Log status
        self.get_logger().info(
            f'IBVS | Image: [{u_current:.1f}, {v_current:.1f}]px, Z={z:.2f}m | '
            f'Err: img={image_error:.1f}px, depth={error_z:.3f}m | '
            f'Cmd: v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f}',
            throttle_duration_sec=0.5
        )

def main(args=None):
    rclpy.init(args=args)
    controller = DemoIBVSController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.cmd_vel_pub.publish(Twist())
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.4 Launch File

Create `demo_visual_servoing.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # Arguments
    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='pbvs',
        description='Controller type: pbvs or ibvs'
    )

    marker_id_arg = DeclareLaunchArgument(
        'marker_id',
        default_value='0',
        description='Target marker ID'
    )

    controller_type = LaunchConfiguration('controller')
    marker_id = LaunchConfiguration('marker_id')

    # PBVS Node
    pbvs_node = Node(
        package='visual_servoing_lab',
        executable='demo_pbvs_controller.py',
        name='pbvs_controller',
        output='screen',
        parameters=[{
            'target_marker_id': marker_id,
            'kp_linear': 0.4,
            'kp_angular': 0.8,
            'desired_distance': 0.5
        }],
        condition=lambda context: context.launch_configurations['controller'] == 'pbvs'
    )

    # IBVS Node
    ibvs_node = Node(
        package='visual_servoing_lab',
        executable='demo_ibvs_controller.py',
        name='ibvs_controller',
        output='screen',
        parameters=[{
            'target_marker_id': marker_id,
            'lambda_gain': 0.5,
            'desired_depth': 0.5
        }],
        condition=lambda context: context.launch_configurations['controller'] == 'ibvs'
    )

    return LaunchDescription([
        controller_arg,
        marker_id_arg,
        pbvs_node,
        ibvs_node
    ])
```

### 3.5 Running the Demo

**Terminal 1 - Start your ArUco TF publisher:**

```bash
ros2 run your_package aruco_pose
```

**Terminal 2 - Run PBVS demo:**

```bash
cd ~/ros2_ws/src/visual_servoing_lab
chmod +x demo_pbvs_controller.py demo_ibvs_controller.py
colcon build
source install/setup.bash

ros2 launch visual_servoing_lab demo_visual_servoing.launch.py controller:=pbvs
```

**Or run IBVS demo:**

```bash
ros2 launch visual_servoing_lab demo_visual_servoing.launch.py controller:=ibvs
```

### 3.6 What the Demo Shows

**PBVS Demo:**

- Reads 3D marker pose from TF
- Computes position error in Cartesian space
- Uses proportional control for linear and angular velocities
- Approaches marker to 50cm while centering it
- Shows clear 3D trajectory

**IBVS Demo:**

- Computes image coordinates from 3D pose
- Controls directly in image space (pixels)
- Adapts to depth changes
- Centers marker in camera view
- Demonstrates image-space control law

---

## Part 4: Testing the Demo

### 4.1 Verify Setup

**Check all systems are running:**

```bash
# Terminal 1: Camera and ArUco TF
ros2 topic list | grep camera
ros2 topic echo /tf --once | grep aruco

# Terminal 2: LiDAR
ros2 topic echo /scan --once

# Terminal 3: Verify transforms
ros2 run tf2_ros tf2_echo camera_link aruco_marker_0
```

### 4.2 Test PBVS Demo

1. **Place robot 1-2 meters from ArUco marker**

2. **Launch PBVS controller:**

   ```bash
   ros2 launch visual_servoing_lab demo_visual_servoing.launch.py controller:=pbvs
   ```

3. **Observe:**

   - Robot moves toward marker in relatively straight path
   - Rotates to face marker
   - Stops at approximately 50cm distance
   - Terminal shows position errors decreasing

4. **Try different parameters:**
   ```bash
   ros2 param set /pbvs_controller kp_linear 0.6
   ros2 param set /pbvs_controller kp_angular 1.2
   ```

### 4.3 Test IBVS Demo

1. **Place robot at different position**

2. **Launch IBVS controller:**

   ```bash
   ros2 launch visual_servoing_lab demo_visual_servoing.launch.py controller:=ibvs
   ```

3. **Observe:**

   - Robot centers marker in camera view
   - Approaches to desired depth
   - May not follow straight 3D path
   - Terminal shows pixel errors decreasing

4. **Monitor in RViz:**
   ```bash
   rviz2
   # Add: TF, Camera, LaserScan
   ```

### 4.4 Visualize Results

**Plot velocity commands (if needed):**

```bash
ros2 topic echo /cmd_vel > velocities.txt
# Plot linear.x and angular.z over time
```

**Record trajectory:**

```bash
ros2 bag record /tf /cmd_vel /scan
```

---

## Part 5: Student Challenges

The following challenges build upon the professor demo. Each challenge focuses on a specific control aspect.

---

### Challenge 1: Pure Angular Alignment Control

**Objective:** Implement a controller that ONLY rotates the robot to face the ArUco marker, without translating.

**Task Description:**
Modify the demo PBVS controller to create a pure angular alignment controller. The robot should:

- Rotate in place to face the marker
- NOT move forward or backward
- Stop when aligned within ±5 degrees

**Requirements:**

1. **Controller Implementation:**

   - Create `angular_alignment_controller.py`
   - Inherit from `MarkerTFListener`
   - Compute angular error to face marker:
     ```python
     # Angle to rotate to face marker
     yaw_to_marker = -atan2(y_marker, z_marker)
     ```
   - Apply ONLY angular velocity (`cmd.angular.z`)
   - Set `cmd.linear.x = 0` always

2. **Control Law:**

   - Implement proportional control: `w = kp * yaw_error`
   - Add velocity limiting: `|w| ≤ 0.5 rad/s`
   - Stop when `|yaw_error| < 0.087 rad` (5 degrees)

3. **Testing:**

   - Test from 5 different starting orientations:
     - 0°, 45°, 90°, 135°, 180° relative to marker
   - Record for each test:
     - Time to alignment
     - Final angular error
     - Maximum angular velocity used

4. **Gain Tuning:**
   - Test with `kp = [0.5, 1.0, 1.5, 2.0]`
   - Plot angular error vs. time for each gain
   - Identify best gain (fastest without oscillation)

**Deliverables:**

- `angular_alignment_controller.py` (working code)
- Test results table (5 orientations × 4 gains)
- 2-3 plots showing angular error convergence
- Brief report (2 pages):
  - Control law explanation
  - Results analysis
  - Best gain justification

**Evaluation Criteria:**

- Correct implementation (no translation) (10 points)
- Proper angular error computation (5 points)
- Comprehensive testing (10 points)
- Report quality (5 points)

---

### Challenge 2: Distance Control with LiDAR Obstacle Avoidance

**Objective:** Implement a controller that approaches the marker to a target distance while using LiDAR to detect and stop before obstacles.

**Task Description:**
Create a controller that moves forward/backward to maintain a desired distance from the marker, but uses LiDAR to ensure safety.

**Requirements:**

1. **Controller Implementation:**

   - Create `distance_controller_with_lidar.py`
   - Inherit from `MarkerTFListener`
   - Subscribe to `/scan` topic (LaserScan messages)
   - Implement distance control in z-direction only
   - Rotate to face marker first, then control distance

2. **LiDAR Processing:**

   ```python
   def process_lidar(self, scan_msg):
       # Get minimum distance in front sector (±30 degrees)
       front_angles = range(len(scan_msg.ranges) // 2 - 30,
                           len(scan_msg.ranges) // 2 + 30)
       front_ranges = [scan_msg.ranges[i] for i in front_angles
                       if scan_msg.range_min < scan_msg.ranges[i] < scan_msg.range_max]

       if front_ranges:
           self.min_obstacle_distance = min(front_ranges)
       else:
           self.min_obstacle_distance = float('inf')
   ```

3. **Safety Logic:**

   - Define safety distance: `d_safe = 0.3m`
   - Define target distance to marker: `d_target = 0.5m`
   - Control law:
     ```python
     if min_obstacle_distance < d_safe:
         v = 0  # STOP
     else:
         error_z = desired_z - current_z
         v = kp * error_z
     ```

4. **Testing Scenarios:**

   - **Scenario A:** Clear path to marker
     - Robot should approach to 0.5m
   - **Scenario B:** Obstacle at 0.4m before marker
     - Robot should stop at 0.3m (safety distance)
   - **Scenario C:** Start too close to marker
     - Robot should back away to 0.5m

5. **Performance Metrics:**
   - Record for each scenario:
     - Final distance to marker
     - Minimum distance to obstacles
     - Safety violations (if any)
     - Convergence time

**Deliverables:**

- `distance_controller_with_lidar.py` (working code)
- Test results for all 3 scenarios
- Plots:
  - Distance to marker vs. time
  - LiDAR minimum distance vs. time
  - Velocity commands vs. time
- Report (3 pages):
  - LiDAR processing method
  - Safety logic explanation
  - Results analysis for each scenario
  - Discussion of safety system effectiveness

**Evaluation Criteria:**

- LiDAR processing implementation (10 points)
- Safety logic correctness (10 points)
- All scenarios tested properly (10 points)
- Report and plots quality (5 points)

---

### Challenge 3: Combined Position Control with Safety Boundaries

**Objective:** Implement BOTH angular and distance control together, with LiDAR-based safety zones and workspace boundaries.

**Task Description:**
Create a complete visual servoing controller that approaches and aligns with the marker while respecting multiple safety constraints.

**Requirements:**

1. **Controller Implementation:**

   - Create `safe_visual_servoing_controller.py`
   - Implement simultaneous angular and distance control
   - Use either PBVS or IBVS approach (your choice)
   - Integrate LiDAR safety monitoring
   - Add workspace boundary enforcement

2. **Control Law:**

   - Compute both linear and angular velocities
   - Example for PBVS:

     ```python
     # Position control
     error_z = desired_z - current_z
     v = kp_linear * error_z

     # Angular control
     yaw_error = -atan2(y, z)
     w = kp_angular * yaw_error
     ```

3. **Safety System:**

   **A. Obstacle Avoidance (LiDAR):**

   - Monitor 180° front sector
   - Three zones:
     - **Critical zone** (< 0.25m): STOP immediately
     - **Warning zone** (0.25-0.40m): Reduce speed by 50%
     - **Safe zone** (> 0.40m): Normal operation

   **B. Workspace Boundaries:**

   - Define bounds on marker distance:
     - `z_min = 0.30m` (don't get too close)
     - `z_max = 2.00m` (don't go too far)
   - Stop if marker outside bounds

   **C. Angular Velocity Limiting:**

   - Maximum angular velocity: `0.8 rad/s`
   - Ramp up/down smoothly (not instant changes)

4. **State Machine:**
   Implement simple state machine:

   ```
   IDLE → ALIGNING → APPROACHING → REACHED → IDLE
   ```

   - **IDLE**: Waiting for marker detection
   - **ALIGNING**: Rotating to face marker (`|yaw_error| > 15°`)
   - **APPROACHING**: Moving to target distance while fine-tuning angle
   - **REACHED**: Goal achieved, hold position

5. **Testing:**
   - **Test 1**: Normal approach (clear path)
   - **Test 2**: Approach with obstacle at 0.35m
   - **Test 3**: Start outside workspace boundary (z > 2m)
   - **Test 4**: Rapid gain changes (tune during operation)

**Deliverables:**

- `safe_visual_servoing_controller.py` (complete implementation)
- State machine diagram
- Test videos or rosbag files for all 4 tests
- Data plots for each test:
  - Robot trajectory (top-down view)
  - Distance to marker vs. time
  - Angular error vs. time
  - LiDAR minimum distance vs. time
  - Velocity commands vs. time
  - State transitions
- Report (4-5 pages):
  - System architecture diagram
  - Control law derivation
  - Safety system detailed explanation
  - State machine description
  - Results and analysis for all tests
  - Discussion of limitations

**Evaluation Criteria:**

- Combined control implementation (10 points)
- LiDAR safety zones correct (8 points)
- Workspace boundaries enforced (5 points)
- State machine implementation (5 points)
- Comprehensive testing (4 points)
- Report quality and depth (3 points)

---

## Part 6: Assessment Rubric

### Challenge Completion (10 points)

- Challenge 1: Angular alignment control (3 points)
- Challenge 2: Distance control with LiDAR (2 points)
- Challenge 3: Combined control with safety (5 points)

**Total: 10 points**

Students must complete at least TWO challenges to pass the laboratory. Completing all three challenges with excellent implementation and reports can earn extra credit.

---

## Part 8: References and Resources

### Essential Reading

1. Chaumette, F., & Hutchinson, S. (2006). "Visual servo control. I. Basic approaches." IEEE Robotics & Automation Magazine, 13(4), 82-90.

2. Chaumette, F., & Hutchinson, S. (2007). "Visual servo control. II. Advanced approaches." IEEE Robotics & Automation Magazine, 14(1), 109-118.

3. Malis, E., Chaumette, F., & Boudet, S. (1999). "2-1/2-D visual servoing." IEEE Transactions on Robotics and Automation, 15(2), 238-250.

4. Corke, P. (2017). "Robotics, Vision and Control: Fundamental Algorithms in MATLAB" (2nd ed.). Springer.

### Additional Resources

- ROS2 Documentation: https://docs.ros.org/
- OpenCV ArUco Documentation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- ViSP Library (Visual Servoing Platform): https://visp.inria.fr/

### Suggested Papers for Challenges

**Challenge 1 (Angular Control):**

- Corke, P. (2017). "Robotics, Vision and Control: Fundamental Algorithms in MATLAB" (2nd ed.), Chapter 15.
- Hutchinson, S., et al. (1996). "A tutorial on visual servo control." IEEE Transactions on Robotics and Automation, 12(5), 651-670.

**Challenge 2 (LiDAR Integration):**

- Borenstein, J., & Koren, Y. (1991). "The vector field histogram - fast obstacle avoidance for mobile robots." IEEE Transactions on Robotics and Automation, 7(3), 278-288.
- Fox, D., et al. (1997). "The dynamic window approach to collision avoidance." IEEE Robotics & Automation Magazine, 4(1), 23-33.

**Challenge 3 (Combined Control with Safety):**

- Konolige, K. (2000). "A gradient method for realtime robot control." IROS 2000.
- Khatib, O. (1986). "Real-time obstacle avoidance for manipulators and mobile robots." The International Journal of Robotics Research, 5(1), 90-98.

---

## Appendix A: Troubleshooting Guide

### Common Issues

**Demo controllers not working:**

- Verify ArUco TF publisher is running: `ros2 topic echo /tf | grep aruco`
- Check camera calibration: `ros2 topic echo /camera/camera_info --once`
- Ensure marker is visible and well-lit
- Verify frame names match: `ros2 run tf2_tools view_frames`

**Robot doesn't move:**

- Check /cmd_vel topic: `ros2 topic echo /cmd_vel`
- Verify motor controller is subscribed: `ros2 topic info /cmd_vel`
- Check for safety limits being triggered
- Ensure marker distance is within reasonable range (0.3m - 2m)

**LiDAR integration issues:**

- Verify scan topic: `ros2 topic echo /scan --once`
- Check scan frame matches TF tree
- Visualize in RViz to see scan data
- Ensure angle ranges are correct for your LiDAR model

**Oscillation or instability:**

- Reduce control gains (kp_linear, kp_angular)
- Add velocity limiting if not present
- Check for control loop timing issues
- Add dead-zone near goal (tolerance region)

### Debug Techniques

1. **Visualize transforms in RViz:**

   ```bash
   rviz2
   # Add: TF, Camera, LaserScan, RobotModel
   ```

2. **Log detailed information:**

   ```python
   self.get_logger().info(f'Position: {position}, Error: {error}')
   ```

3. **Test components separately:**

   - First test TF reading only
   - Then test control law with fixed commands
   - Finally integrate full closed-loop control

4. **Record data for analysis:**
   ```bash
   ros2 bag record /tf /cmd_vel /scan /camera/image_raw
   ```

---

## Appendix B: Extension Ideas

For advanced students who complete all challenges:

1. **Dynamic Obstacle Avoidance:** Extend Challenge 2 to handle moving obstacles detected via LiDAR, requiring real-time trajectory replanning.

2. **Multi-Marker Tracking:** Track multiple markers simultaneously and servo to the centroid or select closest/largest marker dynamically.

3. **Velocity Profiling:** Implement smooth acceleration profiles (S-curves) instead of step changes for more natural robot motion.

4. **Predictive Control:** Add Kalman filtering to predict marker motion and compensate for control delays, improving response time.

5. **Hybrid Approach:** Combine IBVS and PBVS - use PBVS for far distances, switch to IBVS for final precision approach.

6. **Formation Control:** Extend to multiple robots maintaining a formation relative to a marker or relative to each other using visual servoing.

---

**End of Laboratory Guide**

_Good luck with your implementations!_
