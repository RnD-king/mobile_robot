import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, LookupException, TransformException, TransformListener
from visualization_msgs.msg import Marker


GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]
Pose2D = Tuple[float, float, float]


class TurtlebotNavHardwareNode(Node):
    """Hardware-oriented TurtleBot navigation node.

    Mission assumption
    ------------------
    - The robot is placed facing the desired direction.
    - The final goal is a parameterized point (goal_x, goal_y) in the initial robot frame.
    - The robot should move fast, but avoid obstacles using the SLAM OccupancyGrid.

    Planner structure
    -----------------
    /map OccupancyGrid
      -> inflated costmap
      -> local-window A*
      -> line-of-sight shortcut
      -> corner rounding
      -> curvature-regulated pure pursuit
      -> /cmd_vel: linear.x, angular.z
    """

    def __init__(self):
        super().__init__('turtlebot_nav_hardware')

        self._declare_parameters()
        self._load_parameters()

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, map_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, sensor_qos
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.raw_path_pub = self.create_publisher(Path, 'raw_astar_path', 10)
        self.goal_marker_pub = self.create_publisher(Marker, 'goal_marker', 10)
        self.local_target_marker_pub = self.create_publisher(
            Marker, 'local_target_marker', 10
        )
        self.window_marker_pub = self.create_publisher(
            Marker, 'local_window_marker', 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.costmap: Optional[List[float]] = None
        self.blocked: Optional[List[bool]] = None
        self.map_dirty = False

        self.latest_scan: Optional[LaserScan] = None
        self.front_min = math.inf
        self.left_min = math.inf
        self.right_min = math.inf
        self.rear_min = math.inf

        self.start_pose: Optional[Pose2D] = None
        self.mission_yaw: Optional[float] = None
        self.goal_xy: Optional[WorldPoint] = None
        self.path_world: List[WorldPoint] = []
        self.raw_path_world: List[WorldPoint] = []
        self.local_target_xy: Optional[WorldPoint] = None
        self.last_local_target_time = self.get_clock().now()
        self.last_plan_time = self.get_clock().now()
        self.force_replan = True
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.recovery_active = False
        self.recovery_started_time = self.get_clock().now()
        self.recovery_turn_direction = 1.0

        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop
        )

        self.get_logger().info(
            'Fast A* smooth navigation node started. '
            f'Target is ({self.goal_x:.2f}, {self.goal_y:.2f}) in the initial robot frame.'
        )

    # ---------------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------------

    def _declare_parameters(self):
        # ROS interfaces
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')

        # Mission
        # Goal coordinates are expressed in the initial robot frame.
        # x: forward from the robot start pose [m]
        # y: left-positive lateral offset from the robot start pose [m]
        self.declare_parameter('goal_x', 5.8)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.18)
        self.declare_parameter('goal_search_radius', 0.45)

        # Control timing
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('replan_period', 0.35)
        self.declare_parameter('max_expansions', 40000)

        # Occupancy/costmap
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('allow_unknown', True)
        self.declare_parameter('treat_unknown_as_free', True)
        self.declare_parameter('unknown_cost', 1.0)
        self.declare_parameter('inflation_radius', 0.18)
        self.declare_parameter('soft_inflation_radius', 0.32)
        self.declare_parameter('soft_inflation_cost', 6.0)

        # Local planning window in the initial robot frame.
        self.declare_parameter('window_back_margin', 0.35)
        self.declare_parameter('window_goal_margin', 0.35)
        self.declare_parameter('window_half_width', 1.35)
        self.declare_parameter('lateral_hard_limit', 1.45)
        self.declare_parameter('lateral_soft_cost_weight', 1.2)
        self.declare_parameter('unknown_lateral_cost_weight', 0.0)

        # Path smoothing
        self.declare_parameter('shortcut_enabled', True)
        self.declare_parameter('shortcut_clearance_radius', 0.18)
        self.declare_parameter('corner_rounding_enabled', True)
        self.declare_parameter('corner_round_distance', 0.36)
        self.declare_parameter('corner_round_min_angle', 0.22)
        self.declare_parameter('corner_round_samples', 6)
        self.declare_parameter('path_resample_spacing', 0.08)

        # Pure pursuit / speed regulation
        self.declare_parameter('max_linear_speed', 0.38) # 0.28
        self.declare_parameter('min_tracking_speed', 0.14) # 0.10
        self.declare_parameter('max_angular_speed', 2.84)
        self.declare_parameter('lookahead_min', 0.30)
        self.declare_parameter('lookahead_max', 0.70)
        self.declare_parameter('lookahead_speed_gain', 0.90)
        self.declare_parameter('corner_lookahead_min', 0.22)
        self.declare_parameter('corner_lookahead_scan_distance', 0.85)
        self.declare_parameter('corner_lookahead_turn_start', 0.35)
        self.declare_parameter('corner_lookahead_turn_full', 1.15)
        self.declare_parameter('corner_lookahead_min_scale', 0.55)
        self.declare_parameter('curvature_speed_gain', 0.18)
        self.declare_parameter('yaw_slowdown_angle', 2.40)
        self.declare_parameter('goal_slowdown_distance', 0.45)
        self.declare_parameter('cost_speed_gain', 0.08)
        self.declare_parameter('cmd_accel_limit', 1.60) # 1.10
        self.declare_parameter('cmd_angular_accel_limit', 14.0)
        self.declare_parameter('rotate_in_place_angle', 1.05)
        self.declare_parameter('rotate_in_place_gain', 2.60)
        self.declare_parameter('rotate_in_place_min_speed', 0.85)

        # Keep the immediate pure-pursuit target from jumping every replan.
        self.declare_parameter('local_target_hold_time', 0.60)
        self.declare_parameter('local_target_reached_dist', 0.24)
        self.declare_parameter('local_target_max_distance', 0.95)
        self.declare_parameter('local_target_max_yaw', 1.75)
        self.declare_parameter('local_target_switch_angle', 0.70)
        self.declare_parameter('local_target_switch_distance_improvement', 0.45)

        # Laser safety and speed scaling
        self.declare_parameter('front_angle_deg', 35.0)
        self.declare_parameter('emergency_stop_dist', 0.14)
        self.declare_parameter('side_emergency_dist', 0.10)
        self.declare_parameter('rear_emergency_dist', 0.10)
        self.declare_parameter('obstacle_slow_dist', 0.42)
        self.declare_parameter('side_slow_dist', 0.18)
        self.declare_parameter('safety_turn_speed', 1.80)

        # Recovery is only for critical proximity. It stops first, then backs
        # away while turning toward the side with more LiDAR clearance.
        self.declare_parameter('recovery_stop_duration', 0.25)
        self.declare_parameter('recovery_back_duration', 0.75)
        self.declare_parameter('recovery_back_speed', 0.08)
        self.declare_parameter('recovery_turn_speed', 1.45)
        self.declare_parameter('recovery_clear_front_dist', 0.24)
        self.declare_parameter('recovery_clear_side_dist', 0.15)

    def _load_parameters(self):
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.fallback_base_frame = str(
            self.get_parameter('fallback_base_frame').value
        )

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.target_distance = math.hypot(self.goal_x, self.goal_y)
        if self.target_distance < 1.0e-3:
            self.get_logger().warn(
                'goal_x and goal_y are almost zero. Forcing goal_x=0.1 m.'
            )
            self.goal_x = 0.1
            self.goal_y = 0.0
            self.target_distance = 0.1
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.goal_search_radius = float(self.get_parameter('goal_search_radius').value)

        self.control_frequency = float(self.get_parameter('control_frequency').value)
        self.replan_period = float(self.get_parameter('replan_period').value)
        self.max_expansions = int(self.get_parameter('max_expansions').value)

        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.allow_unknown = bool(self.get_parameter('allow_unknown').value)
        self.treat_unknown_as_free = bool(
            self.get_parameter('treat_unknown_as_free').value
        )
        self.unknown_cost = float(self.get_parameter('unknown_cost').value)
        self.inflation_radius = float(self.get_parameter('inflation_radius').value)
        self.soft_inflation_radius = float(
            self.get_parameter('soft_inflation_radius').value
        )
        self.soft_inflation_cost = float(
            self.get_parameter('soft_inflation_cost').value
        )

        self.window_back_margin = float(
            self.get_parameter('window_back_margin').value
        )
        self.window_goal_margin = float(
            self.get_parameter('window_goal_margin').value
        )
        self.window_half_width = float(self.get_parameter('window_half_width').value)
        self.lateral_hard_limit = float(
            self.get_parameter('lateral_hard_limit').value
        )
        self.lateral_soft_cost_weight = float(
            self.get_parameter('lateral_soft_cost_weight').value
        )
        self.unknown_lateral_cost_weight = float(
            self.get_parameter('unknown_lateral_cost_weight').value
        )

        self.shortcut_enabled = bool(self.get_parameter('shortcut_enabled').value)
        self.shortcut_clearance_radius = float(
            self.get_parameter('shortcut_clearance_radius').value
        )
        self.corner_rounding_enabled = bool(
            self.get_parameter('corner_rounding_enabled').value
        )
        self.corner_round_distance = float(
            self.get_parameter('corner_round_distance').value
        )
        self.corner_round_min_angle = float(
            self.get_parameter('corner_round_min_angle').value
        )
        self.corner_round_samples = int(
            self.get_parameter('corner_round_samples').value
        )
        self.path_resample_spacing = float(
            self.get_parameter('path_resample_spacing').value
        )

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.min_tracking_speed = float(
            self.get_parameter('min_tracking_speed').value
        )
        self.max_angular_speed = float(
            self.get_parameter('max_angular_speed').value
        )
        self.lookahead_min = float(self.get_parameter('lookahead_min').value)
        self.lookahead_max = float(self.get_parameter('lookahead_max').value)
        self.lookahead_speed_gain = float(
            self.get_parameter('lookahead_speed_gain').value
        )
        self.corner_lookahead_min = float(
            self.get_parameter('corner_lookahead_min').value
        )
        self.corner_lookahead_scan_distance = float(
            self.get_parameter('corner_lookahead_scan_distance').value
        )
        self.corner_lookahead_turn_start = float(
            self.get_parameter('corner_lookahead_turn_start').value
        )
        self.corner_lookahead_turn_full = float(
            self.get_parameter('corner_lookahead_turn_full').value
        )
        self.corner_lookahead_min_scale = float(
            self.get_parameter('corner_lookahead_min_scale').value
        )
        self.curvature_speed_gain = float(
            self.get_parameter('curvature_speed_gain').value
        )
        self.yaw_slowdown_angle = float(
            self.get_parameter('yaw_slowdown_angle').value
        )
        self.goal_slowdown_distance = float(
            self.get_parameter('goal_slowdown_distance').value
        )
        self.cost_speed_gain = float(self.get_parameter('cost_speed_gain').value)
        self.cmd_accel_limit = float(self.get_parameter('cmd_accel_limit').value)
        self.cmd_angular_accel_limit = float(
            self.get_parameter('cmd_angular_accel_limit').value
        )
        self.rotate_in_place_angle = float(
            self.get_parameter('rotate_in_place_angle').value
        )
        self.rotate_in_place_gain = float(
            self.get_parameter('rotate_in_place_gain').value
        )
        self.rotate_in_place_min_speed = float(
            self.get_parameter('rotate_in_place_min_speed').value
        )
        self.local_target_hold_time = float(
            self.get_parameter('local_target_hold_time').value
        )
        self.local_target_reached_dist = float(
            self.get_parameter('local_target_reached_dist').value
        )
        self.local_target_max_distance = float(
            self.get_parameter('local_target_max_distance').value
        )
        self.local_target_max_yaw = float(
            self.get_parameter('local_target_max_yaw').value
        )
        self.local_target_switch_angle = float(
            self.get_parameter('local_target_switch_angle').value
        )
        self.local_target_switch_distance_improvement = float(
            self.get_parameter('local_target_switch_distance_improvement').value
        )

        self.front_angle_deg = float(self.get_parameter('front_angle_deg').value)
        self.emergency_stop_dist = float(
            self.get_parameter('emergency_stop_dist').value
        )
        self.side_emergency_dist = float(
            self.get_parameter('side_emergency_dist').value
        )
        self.rear_emergency_dist = float(
            self.get_parameter('rear_emergency_dist').value
        )
        self.obstacle_slow_dist = float(
            self.get_parameter('obstacle_slow_dist').value
        )
        self.side_slow_dist = float(self.get_parameter('side_slow_dist').value)
        self.safety_turn_speed = float(self.get_parameter('safety_turn_speed').value)
        self.recovery_stop_duration = float(
            self.get_parameter('recovery_stop_duration').value
        )
        self.recovery_back_duration = float(
            self.get_parameter('recovery_back_duration').value
        )
        self.recovery_back_speed = float(
            self.get_parameter('recovery_back_speed').value
        )
        self.recovery_turn_speed = float(
            self.get_parameter('recovery_turn_speed').value
        )
        self.recovery_clear_front_dist = float(
            self.get_parameter('recovery_clear_front_dist').value
        )
        self.recovery_clear_side_dist = float(
            self.get_parameter('recovery_clear_side_dist').value
        )

    # ---------------------------------------------------------------------
    # ROS callbacks
    # ---------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.costmap, self.blocked = self.build_inflated_costmap(msg)
        self.map_dirty = True

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.front_min, self.left_min, self.right_min, self.rear_min = (
            self.compute_sector_min_distances(msg)
        )

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def control_loop(self):
        if self.map_msg is None or self.costmap is None or self.blocked is None:
            self.stop_robot()
            return

        pose = self.lookup_robot_pose()
        if pose is None:
            self.stop_robot()
            return

        if self.start_pose is None:
            self.initialize_mission(pose)
            self.stop_robot()
            return

        robot_x, robot_y, robot_yaw = pose
        robot_xy = (robot_x, robot_y)
        goal_distance = self.distance(robot_xy, self.goal_xy)

        self.publish_goal_marker()
        self.publish_window_marker(robot_xy)

        if goal_distance <= self.goal_tolerance:
            self.get_logger().info('Goal reached. Stopping robot.', throttle_duration_sec=1.0)
            self.path_world = []
            self.raw_path_world = []
            self.local_target_xy = None
            self.publish_path([], self.path_pub)
            self.publish_path([], self.raw_path_pub)
            self.stop_robot()
            return

        if self.recovery_active:
            recovery_cmd = self.compute_recovery_command()
            if recovery_cmd is not None:
                self.cmd_pub.publish(recovery_cmd)
                return

        if self.is_hard_safety_stop():
            self.start_recovery('critical laser clearance')
            recovery_cmd = self.compute_recovery_command()
            if recovery_cmd is not None:
                self.cmd_pub.publish(recovery_cmd)
            return

        if self.need_replan(robot_xy):
            if not self.plan_path(robot_xy):
                self.get_logger().warn(
                    'Planning failed. Stopping and waiting for map update.',
                    throttle_duration_sec=1.0,
                )
                self.stop_robot()
                return

        cmd = self.compute_pure_pursuit_command(pose)
        self.cmd_pub.publish(cmd)

    def initialize_mission(self, pose: Pose2D):
        x, y, yaw = pose
        self.start_pose = (x, y, yaw)

        # goal_x and goal_y are given in the initial robot frame.
        # The planning window is aligned with the straight line from the
        # start pose to that goal point, so lateral limits mean
        # "distance away from the start-goal line."
        self.mission_yaw = yaw + math.atan2(self.goal_y, self.goal_x)
        self.goal_xy = self.from_start_frame(self.target_distance, 0.0)

        self.force_replan = True
        self.last_plan_time = self.get_clock().now()
        self.get_logger().info(
            'Mission initialized. '
            f'start=({x:.3f}, {y:.3f}, yaw={yaw:.3f}), '
            f'relative_goal=({self.goal_x:.3f}, {self.goal_y:.3f}), '
            f'mission_yaw={self.mission_yaw:.3f}, '
            f'map_goal=({self.goal_xy[0]:.3f}, {self.goal_xy[1]:.3f})'
        )

    # ---------------------------------------------------------------------
    # TF and geometry
    # ---------------------------------------------------------------------

    def lookup_robot_pose(self) -> Optional[Pose2D]:
        for base in (self.base_frame, self.fallback_base_frame):
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    base,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.05),
                )
                q = trans.transform.rotation
                yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
                return (
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    yaw,
                )
            except (LookupException, TransformException):
                continue

        self.get_logger().warn(
            f'Cannot lookup TF {self.map_frame}->{self.base_frame}.',
            throttle_duration_sec=2.0,
        )
        return None

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def distance(a: WorldPoint, b: WorldPoint) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def elapsed_seconds(self, now, past) -> float:
        return max(0.0, (now.nanoseconds - past.nanoseconds) / 1.0e9)

    def to_start_frame(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy, _ = self.start_pose
        frame_yaw = self.mission_yaw if self.mission_yaw is not None else self.start_pose[2]
        dx = x - sx
        dy = y - sy
        c = math.cos(frame_yaw)
        s = math.sin(frame_yaw)
        forward = dx * c + dy * s
        lateral = -dx * s + dy * c
        return forward, lateral

    def from_start_frame(self, forward: float, lateral: float) -> WorldPoint:
        sx, sy, _ = self.start_pose
        frame_yaw = self.mission_yaw if self.mission_yaw is not None else self.start_pose[2]
        c = math.cos(frame_yaw)
        s = math.sin(frame_yaw)
        x = sx + forward * c - lateral * s
        y = sy + forward * s + lateral * c
        return x, y

    # ---------------------------------------------------------------------
    # OccupancyGrid utilities
    # ---------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Optional[GridCell]:
        cell = self.world_to_grid_unbounded(x, y)
        if not self.is_inside_grid(*cell):
            return None
        return cell

    def world_to_grid_unbounded(self, x: float, y: float) -> GridCell:
        info = self.map_msg.info
        gx = int(math.floor((x - info.origin.position.x) / info.resolution))
        gy = int(math.floor((y - info.origin.position.y) / info.resolution))
        return gx, gy

    def grid_to_world(self, cell: GridCell) -> WorldPoint:
        info = self.map_msg.info
        gx, gy = cell
        x = info.origin.position.x + (gx + 0.5) * info.resolution
        y = info.origin.position.y + (gy + 0.5) * info.resolution
        return x, y

    def is_inside_grid(self, gx: int, gy: int) -> bool:
        info = self.map_msg.info
        return 0 <= gx < info.width and 0 <= gy < info.height

    def grid_index(self, cell: GridCell) -> int:
        gx, gy = cell
        return gy * self.map_msg.info.width + gx

    def build_inflated_costmap(
        self, msg: OccupancyGrid
    ) -> Tuple[List[float], List[bool]]:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        total = width * height

        costmap = [1.0] * total
        blocked = [False] * total
        obstacle_cells: List[GridCell] = []

        for idx, value in enumerate(msg.data):
            if value >= self.occupied_threshold:
                blocked[idx] = True
                obstacle_cells.append((idx % width, idx // width))
            elif value < 0:
                if self.treat_unknown_as_free:
                    costmap[idx] = 1.0
                elif self.allow_unknown:
                    costmap[idx] = self.unknown_cost
                else:
                    blocked[idx] = True
            else:
                costmap[idx] = 1.0

        hard_cells = int(math.ceil(self.inflation_radius / resolution))
        soft_radius = max(self.soft_inflation_radius, self.inflation_radius)
        soft_cells = int(math.ceil(soft_radius / resolution))
        if soft_cells <= 0:
            return costmap, blocked

        offsets = self.make_inflation_offsets(soft_cells)
        for ox, oy in obstacle_cells:
            for dx, dy in offsets:
                nx = ox + dx
                ny = oy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                dist_cells = math.hypot(dx, dy)
                idx = ny * width + nx
                if dist_cells <= hard_cells:
                    blocked[idx] = True
                else:
                    denom = max(1.0, soft_cells - hard_cells)
                    ratio = max(0.0, 1.0 - (dist_cells - hard_cells) / denom)
                    costmap[idx] = max(
                        costmap[idx],
                        1.0 + self.soft_inflation_cost * ratio,
                    )

        return costmap, blocked

    @staticmethod
    def make_inflation_offsets(radius_cells: int) -> List[GridCell]:
        offsets: List[GridCell] = []
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dx, dy))
        return offsets

    def is_cell_in_local_window(self, cell: GridCell) -> bool:
        if self.start_pose is None:
            return True
        x, y = self.grid_to_world(cell)
        forward, lateral = self.to_start_frame(x, y)
        min_forward = -self.window_back_margin
        max_forward = self.target_distance + self.window_goal_margin
        lateral_limit = min(self.window_half_width, self.lateral_hard_limit)
        return (
            min_forward <= forward <= max_forward
            and abs(lateral) <= lateral_limit
        )

    def is_plannable_cell(self, cell: GridCell) -> bool:
        if not self.is_inside_grid(*cell):
            return False
        if not self.is_cell_in_local_window(cell):
            return False
        if self.blocked is None:
            return False
        return not self.blocked[self.grid_index(cell)]

    def cell_traversal_cost(self, cell: GridCell) -> float:
        if not self.is_plannable_cell(cell):
            return math.inf
        idx = self.grid_index(cell)
        cost = self.costmap[idx]

        x, y = self.grid_to_world(cell)
        _, lateral = self.to_start_frame(x, y)
        norm_lat = min(1.0, abs(lateral) / max(0.05, self.window_half_width))
        cost += self.lateral_soft_cost_weight * norm_lat * norm_lat

        if self.map_msg.data[idx] < 0 and not self.treat_unknown_as_free:
            cost += self.unknown_lateral_cost_weight * norm_lat

        return cost

    def nearest_plannable_cell(
        self, cell: GridCell, max_radius_m: float
    ) -> Optional[GridCell]:
        if self.is_plannable_cell(cell):
            return cell

        max_radius = max(1, int(math.ceil(max_radius_m / self.map_msg.info.resolution)))
        cx, cy = cell
        best_cell = None
        best_dist = math.inf

        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cx + dx, cy + dy)
                    if not self.is_plannable_cell(candidate):
                        continue
                    x, y = self.grid_to_world(candidate)
                    _, lateral = self.to_start_frame(x, y)
                    dist = math.hypot(dx, dy) + 0.2 * abs(lateral)
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = candidate
            if best_cell is not None:
                return best_cell
        return None

    # ---------------------------------------------------------------------
    # A* planning
    # ---------------------------------------------------------------------

    def need_replan(self, robot_xy: WorldPoint) -> bool:
        now = self.get_clock().now()
        if self.force_replan:
            return True
        if not self.path_world:
            return True
        if self.elapsed_seconds(now, self.last_plan_time) >= self.replan_period:
            return True
        if self.map_dirty and self.is_path_blocked(robot_xy):
            return True
        return False

    def plan_path(self, robot_xy: WorldPoint) -> bool:
        self.last_plan_time = self.get_clock().now()
        self.force_replan = False
        self.map_dirty = False

        start_cell = self.world_to_grid_unbounded(robot_xy[0], robot_xy[1])
        start_cell = self.nearest_plannable_cell(start_cell, self.goal_search_radius)
        if start_cell is None:
            self.get_logger().warn('No plannable start cell found.', throttle_duration_sec=1.0)
            return False

        goal_cell = self.select_goal_cell(robot_xy)
        if goal_cell is None:
            self.get_logger().warn('No plannable goal cell found.', throttle_duration_sec=1.0)
            return False

        raw_grid_path = self.astar(start_cell, goal_cell)
        if not raw_grid_path:
            self.get_logger().warn('A* failed inside local window.', throttle_duration_sec=1.0)
            return False

        self.raw_path_world = [self.grid_to_world(cell) for cell in raw_grid_path]
        shortcut_path = self.shortcut_grid_path(raw_grid_path)
        path_world = [self.grid_to_world(cell) for cell in shortcut_path]
        path_world = self.round_corners(path_world)
        path_world = self.resample_path(path_world, self.path_resample_spacing)

        if len(path_world) < 2:
            path_world = self.raw_path_world

        self.path_world = path_world
        self.publish_path(self.raw_path_world, self.raw_path_pub)
        self.publish_path(self.path_world, self.path_pub)

        self.get_logger().info(
            f'Planned path: raw={len(self.raw_path_world)} points, '
            f'smooth={len(self.path_world)} points.',
            throttle_duration_sec=0.8,
        )
        return True

    def select_goal_cell(self, robot_xy: WorldPoint) -> Optional[GridCell]:
        final_goal = self.world_to_grid_unbounded(self.goal_xy[0], self.goal_xy[1])
        nearest_final = self.nearest_plannable_cell(final_goal, self.goal_search_radius)
        if nearest_final is not None:
            return nearest_final

        # If the final goal is outside the current map, select the farthest
        # plannable cell near the center line. This is not frontier exploration;
        # it is just a temporary local target until the final goal becomes mapped.
        robot_s, _ = self.to_start_frame(robot_xy[0], robot_xy[1])
        resolution = self.map_msg.info.resolution
        best_cell = None
        best_score = -math.inf

        s_start = max(robot_s + 0.4, 0.0)
        s_end = self.target_distance
        lateral_samples = self.make_lateral_samples(self.window_half_width, resolution)
        steps = max(1, int(math.ceil((s_end - s_start) / resolution)))

        for i in range(steps + 1):
            s = s_start + i * resolution
            for d in lateral_samples:
                x, y = self.from_start_frame(s, d)
                cell = self.world_to_grid_unbounded(x, y)
                if not self.is_plannable_cell(cell):
                    continue
                # Prefer forward progress, then prefer the center line.
                score = s - 0.35 * abs(d)
                if score > best_score:
                    best_score = score
                    best_cell = cell

        return best_cell

    @staticmethod
    def make_lateral_samples(half_width: float, resolution: float) -> List[float]:
        samples = [0.0]
        max_idx = max(1, int(math.ceil(half_width / resolution)))
        for i in range(1, max_idx + 1):
            d = i * resolution
            samples.append(d)
            samples.append(-d)
        return samples

    def astar(self, start: GridCell, goal: GridCell) -> List[GridCell]:
        open_heap: List[Tuple[float, GridCell]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start: 0.0}
        closed: Set[GridCell] = set()
        expansions = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed.add(current)
            expansions += 1
            if expansions > self.max_expansions:
                self.get_logger().warn('A* expansion limit reached.', throttle_duration_sec=1.0)
                return []

            for neighbor, step_cost in self.get_neighbors(current):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def get_neighbors(self, cell: GridCell) -> List[Tuple[GridCell, float]]:
        gx, gy = cell
        result: List[Tuple[GridCell, float]] = []
        directions = (
            (-1, 0, 1.0), (1, 0, 1.0),
            (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )

        for dx, dy, move_cost in directions:
            neighbor = (gx + dx, gy + dy)
            if not self.is_plannable_cell(neighbor):
                continue

            # Prevent diagonal squeezing through two blocked corner cells.
            if dx != 0 and dy != 0:
                side_a = (gx + dx, gy)
                side_b = (gx, gy + dy)
                if not self.is_plannable_cell(side_a) or not self.is_plannable_cell(side_b):
                    continue

            traversal_cost = self.cell_traversal_cost(neighbor)
            if not math.isfinite(traversal_cost):
                continue
            result.append((neighbor, move_cost * traversal_cost))

        return result

    @staticmethod
    def heuristic(a: GridCell, b: GridCell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def reconstruct_path(
        came_from: Dict[GridCell, GridCell], current: GridCell
    ) -> List[GridCell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ---------------------------------------------------------------------
    # Path smoothing
    # ---------------------------------------------------------------------

    def shortcut_grid_path(self, path: List[GridCell]) -> List[GridCell]:
        if not self.shortcut_enabled or len(path) <= 2:
            return path

        clearance_cells = max(
            0,
            int(math.ceil(self.shortcut_clearance_radius / self.map_msg.info.resolution)),
        )

        smoothed = [path[0]]
        anchor_idx = 0
        while anchor_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > anchor_idx + 1:
                if self.has_line_of_sight(path[anchor_idx], path[next_idx], clearance_cells):
                    break
                next_idx -= 1
            smoothed.append(path[next_idx])
            anchor_idx = next_idx
        return smoothed

    def has_line_of_sight(
        self, start: GridCell, end: GridCell, clearance_cells: int = 0
    ) -> bool:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x = x0
        y = y0

        while True:
            if not self.has_cell_clearance((x, y), clearance_cells):
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def has_cell_clearance(self, cell: GridCell, clearance_cells: int) -> bool:
        if clearance_cells <= 0:
            return self.is_plannable_cell(cell)

        cx, cy = cell
        r2 = clearance_cells * clearance_cells
        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):
                if dx * dx + dy * dy > r2:
                    continue
                if not self.is_plannable_cell((cx + dx, cy + dy)):
                    return False
        return True

    def round_corners(self, path: List[WorldPoint]) -> List[WorldPoint]:
        if not self.corner_rounding_enabled or len(path) <= 2:
            return path

        rounded: List[WorldPoint] = [path[0]]
        for i in range(1, len(path) - 1):
            p0 = path[i - 1]
            p1 = path[i]
            p2 = path[i + 1]

            v_in = (p1[0] - p0[0], p1[1] - p0[1])
            v_out = (p2[0] - p1[0], p2[1] - p1[1])
            len_in = math.hypot(v_in[0], v_in[1])
            len_out = math.hypot(v_out[0], v_out[1])
            if len_in < 1.0e-6 or len_out < 1.0e-6:
                continue

            u_in = (v_in[0] / len_in, v_in[1] / len_in)
            u_out = (v_out[0] / len_out, v_out[1] / len_out)
            dot = self.clamp(u_in[0] * u_out[0] + u_in[1] * u_out[1], -1.0, 1.0)
            turn_angle = math.acos(dot)

            if turn_angle < self.corner_round_min_angle:
                rounded.append(p1)
                continue

            cut = min(self.corner_round_distance, 0.45 * len_in, 0.45 * len_out)
            if cut < 0.03:
                rounded.append(p1)
                continue

            a = (p1[0] - u_in[0] * cut, p1[1] - u_in[1] * cut)
            b = (p1[0] + u_out[0] * cut, p1[1] + u_out[1] * cut)
            curve = self.quadratic_bezier_points(a, p1, b, self.corner_round_samples)

            if self.curve_is_clear(curve):
                if self.distance(rounded[-1], a) > 1.0e-4:
                    rounded.append(a)
                rounded.extend(curve[1:])
            else:
                rounded.append(p1)

        rounded.append(path[-1])
        return rounded

    @staticmethod
    def quadratic_bezier_points(
        a: WorldPoint, c: WorldPoint, b: WorldPoint, samples: int
    ) -> List[WorldPoint]:
        n = max(2, samples)
        points: List[WorldPoint] = []
        for i in range(n + 1):
            t = i / n
            mt = 1.0 - t
            x = mt * mt * a[0] + 2.0 * mt * t * c[0] + t * t * b[0]
            y = mt * mt * a[1] + 2.0 * mt * t * c[1] + t * t * b[1]
            points.append((x, y))
        return points

    def curve_is_clear(self, points: List[WorldPoint]) -> bool:
        clearance_cells = max(
            0,
            int(math.ceil(self.shortcut_clearance_radius / self.map_msg.info.resolution)),
        )
        for point in points:
            cell = self.world_to_grid(point[0], point[1])
            if cell is None:
                return False
            if not self.has_cell_clearance(cell, clearance_cells):
                return False
        return True

    def resample_path(self, path: List[WorldPoint], spacing: float) -> List[WorldPoint]:
        if len(path) <= 2 or spacing <= 1.0e-4:
            return path

        result: List[WorldPoint] = [path[0]]
        for i in range(1, len(path)):
            p0 = result[-1]
            p1 = path[i]
            seg_len = self.distance(p0, p1)
            if seg_len < spacing:
                continue
            steps = max(1, int(math.floor(seg_len / spacing)))
            for k in range(1, steps + 1):
                t = min(1.0, k * spacing / seg_len)
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                result.append((x, y))
        if self.distance(result[-1], path[-1]) > 1.0e-4:
            result.append(path[-1])
        return result

    def is_path_blocked(self, robot_xy: WorldPoint) -> bool:
        if not self.path_world:
            return True
        clearance_cells = max(
            0,
            int(math.ceil(self.shortcut_clearance_radius / self.map_msg.info.resolution)),
        )
        checked = 0
        for point in self.path_world:
            if self.distance(robot_xy, point) < self.lookahead_min * 0.5:
                continue
            cell = self.world_to_grid(point[0], point[1])
            if cell is None or not self.has_cell_clearance(cell, clearance_cells):
                return True
            checked += 1
            if checked >= 40:
                break
        return False

    # ---------------------------------------------------------------------
    # Pure pursuit tracking
    # ---------------------------------------------------------------------

    def compute_pure_pursuit_command(self, pose: Pose2D) -> Twist:
        robot_x, robot_y, robot_yaw = pose
        cmd = Twist()

        target = self.select_lookahead_point(robot_x, robot_y, robot_yaw)
        if target is None:
            return cmd

        self.publish_local_target_marker(target)

        dx = target[0] - robot_x
        dy = target[1] - robot_y
        lookahead = max(0.05, math.hypot(dx, dy))
        target_yaw = math.atan2(dy, dx)
        alpha = self.normalize_angle(target_yaw - robot_yaw)

        if abs(alpha) >= self.rotate_in_place_angle:
            return self.compute_rotate_in_place_command(alpha)

        curvature = 2.0 * math.sin(alpha) / lookahead
        curvature_abs = abs(curvature)

        curvature_scale = 1.0 / (1.0 + self.curvature_speed_gain * curvature_abs)
        yaw_scale = max(0.0, 1.0 - abs(alpha) / max(0.1, self.yaw_slowdown_angle))
        yaw_scale = max(0.55, yaw_scale)
        obstacle_scale = min(self.front_obstacle_speed_scale(), self.side_obstacle_speed_scale())
        cost_scale = self.local_cost_speed_scale(target)
        goal_scale = self.goal_speed_scale((robot_x, robot_y))

        v = (
            self.max_linear_speed
            * curvature_scale
            * yaw_scale
            * obstacle_scale
            * cost_scale
            * goal_scale
        )

        if v > 0.0:
            v = max(self.min_tracking_speed, v)

        # Respect angular velocity limit. Since kappa = omega / v,
        # v must be reduced when the requested curvature is too large.
        if curvature_abs > 1.0e-4:
            v = min(v, self.max_angular_speed / curvature_abs)

        omega = v * curvature
        omega = self.clamp(omega, -self.max_angular_speed, self.max_angular_speed)

        v, omega = self.apply_command_accel_limits(v, omega)

        cmd.linear.x = v
        cmd.angular.z = omega
        self.last_cmd_v = v
        self.last_cmd_w = omega
        return cmd

    def compute_rotate_in_place_command(self, yaw_error: float) -> Twist:
        cmd = Twist()
        omega = self.clamp(
            self.rotate_in_place_gain * yaw_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        if abs(omega) < self.rotate_in_place_min_speed:
            omega = math.copysign(self.rotate_in_place_min_speed, yaw_error)

        omega = self.apply_angular_accel_limit(omega)
        cmd.linear.x = 0.0
        cmd.angular.z = omega
        self.last_cmd_v = 0.0
        self.last_cmd_w = omega
        return cmd

    def select_lookahead_point(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ) -> Optional[WorldPoint]:
        if not self.path_world:
            return None

        robot = (robot_x, robot_y)
        nearest_idx = 0
        nearest_dist = math.inf
        for idx, point in enumerate(self.path_world):
            dist = self.distance(robot, point)
            if dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist

        if nearest_idx > 0:
            self.path_world = self.path_world[nearest_idx:]

        lookahead = self.compute_adaptive_lookahead()

        candidate = None
        path_distance = self.distance(robot, self.path_world[0])
        prev_point = self.path_world[0]
        for idx, point in enumerate(self.path_world):
            if idx > 0:
                path_distance += self.distance(prev_point, point)
                prev_point = point
            if path_distance < lookahead:
                continue
            angle = math.atan2(point[1] - robot_y, point[0] - robot_x)
            yaw_error = abs(self.normalize_angle(angle - robot_yaw))
            if yaw_error <= self.local_target_max_yaw:
                candidate = point
                break

        if candidate is None:
            candidate = self.path_world[-1]

        target = self.stabilize_local_target(robot, robot_yaw, candidate)
        return target

    def compute_adaptive_lookahead(self) -> float:
        base = self.lookahead_min + self.lookahead_speed_gain * abs(self.last_cmd_v)
        base = self.clamp(base, self.lookahead_min, self.lookahead_max)

        upcoming_turn = self.estimate_upcoming_path_turn()
        span = max(
            0.01,
            self.corner_lookahead_turn_full - self.corner_lookahead_turn_start,
        )
        turn_ratio = self.clamp(
            (upcoming_turn - self.corner_lookahead_turn_start) / span,
            0.0,
            1.0,
        )
        scale = 1.0 - turn_ratio * (1.0 - self.corner_lookahead_min_scale)
        return self.clamp(
            base * scale,
            self.corner_lookahead_min,
            self.lookahead_max,
        )

    def estimate_upcoming_path_turn(self) -> float:
        if len(self.path_world) < 3:
            return 0.0

        travelled = 0.0
        total_turn = 0.0
        last_heading = None

        for i in range(len(self.path_world) - 1):
            p0 = self.path_world[i]
            p1 = self.path_world[i + 1]
            seg_len = self.distance(p0, p1)
            if seg_len < 1.0e-4:
                continue

            heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            if last_heading is not None:
                total_turn += abs(self.normalize_angle(heading - last_heading))
            last_heading = heading

            travelled += seg_len
            if travelled >= self.corner_lookahead_scan_distance:
                break

        return min(math.pi, total_turn)

    def stabilize_local_target(
        self, robot: WorldPoint, robot_yaw: float, candidate: WorldPoint
    ) -> WorldPoint:
        previous = self.local_target_xy
        if previous is None:
            self.local_target_xy = candidate
            self.last_local_target_time = self.get_clock().now()
            return candidate

        if not self.is_local_target_still_usable(robot, robot_yaw, previous):
            self.local_target_xy = candidate
            self.last_local_target_time = self.get_clock().now()
            return candidate

        prev_angle = math.atan2(previous[1] - robot[1], previous[0] - robot[0])
        cand_angle = math.atan2(candidate[1] - robot[1], candidate[0] - robot[0])
        direction_change = abs(self.normalize_angle(cand_angle - prev_angle))
        target_jump = self.distance(previous, candidate)

        if (
            direction_change > self.local_target_switch_angle
            and target_jump > self.local_target_reached_dist
            and not self.is_new_target_much_better(robot, previous, candidate)
        ):
            return previous

        if target_jump > 0.04:
            self.local_target_xy = candidate
            self.last_local_target_time = self.get_clock().now()
        return self.local_target_xy

    def is_new_target_much_better(
        self, robot: WorldPoint, previous: WorldPoint, candidate: WorldPoint
    ) -> bool:
        if self.goal_xy is None:
            return False

        previous_estimate = (
            self.distance(robot, previous)
            + self.distance(previous, self.goal_xy)
        )
        candidate_estimate = (
            self.distance(robot, candidate)
            + self.distance(candidate, self.goal_xy)
        )
        improvement = previous_estimate - candidate_estimate
        return improvement >= self.local_target_switch_distance_improvement

    def is_local_target_still_usable(
        self, robot: WorldPoint, robot_yaw: float, target: WorldPoint
    ) -> bool:
        dist = self.distance(robot, target)
        if dist < self.local_target_reached_dist:
            return False
        if dist > self.local_target_max_distance:
            return False

        target_yaw = math.atan2(target[1] - robot[1], target[0] - robot[0])
        yaw_error = abs(self.normalize_angle(target_yaw - robot_yaw))
        if yaw_error > self.local_target_max_yaw:
            return False

        cell = self.world_to_grid(target[0], target[1])
        if cell is None:
            return False
        return self.is_plannable_cell(cell)

    def front_obstacle_speed_scale(self) -> float:
        if not math.isfinite(self.front_min):
            return 1.0
        if self.front_min <= self.emergency_stop_dist:
            return 0.0
        if self.front_min >= self.obstacle_slow_dist:
            return 1.0
        span = max(0.01, self.obstacle_slow_dist - self.emergency_stop_dist)
        ratio = (self.front_min - self.emergency_stop_dist) / span
        return self.clamp(ratio, 0.35, 1.0)

    def side_obstacle_speed_scale(self) -> float:
        side_min = min(self.left_min, self.right_min)
        if not math.isfinite(side_min):
            return 1.0
        if side_min <= self.side_emergency_dist:
            return 0.0
        if side_min >= self.side_slow_dist:
            return 1.0
        span = max(0.01, self.side_slow_dist - self.side_emergency_dist)
        ratio = (side_min - self.side_emergency_dist) / span
        return self.clamp(ratio, 0.35, 1.0)

    def local_cost_speed_scale(self, target: WorldPoint) -> float:
        cell = self.world_to_grid(target[0], target[1])
        if cell is None or self.costmap is None:
            return 0.5
        cost = self.cell_traversal_cost(cell)
        if not math.isfinite(cost):
            return 0.25
        if cost <= 1.0:
            return 1.0
        return self.clamp(1.0 / (1.0 + self.cost_speed_gain * (cost - 1.0)), 0.45, 1.0)

    def goal_speed_scale(self, robot_xy: WorldPoint) -> float:
        dist = self.distance(robot_xy, self.goal_xy)
        if dist >= self.goal_slowdown_distance:
            return 1.0
        return self.clamp(dist / max(0.05, self.goal_slowdown_distance), 0.35, 1.0)

    def apply_command_accel_limits(self, v: float, omega: float) -> Tuple[float, float]:
        dt = 1.0 / max(1.0, self.control_frequency)
        max_dv = self.cmd_accel_limit * dt
        max_dw = self.cmd_angular_accel_limit * dt
        v = self.clamp(v, self.last_cmd_v - max_dv, self.last_cmd_v + max_dv)
        omega = self.clamp(omega, self.last_cmd_w - max_dw, self.last_cmd_w + max_dw)
        return v, omega

    def apply_angular_accel_limit(self, omega: float) -> float:
        dt = 1.0 / max(1.0, self.control_frequency)
        max_dw = self.cmd_angular_accel_limit * dt
        return self.clamp(omega, self.last_cmd_w - max_dw, self.last_cmd_w + max_dw)

    # ---------------------------------------------------------------------
    # Laser safety layer
    # ---------------------------------------------------------------------

    def compute_sector_min_distances(
        self, msg: LaserScan
    ) -> Tuple[float, float, float, float]:
        front_half_angle = math.radians(self.front_angle_deg) * 0.5
        side_half_angle = math.radians(45.0)
        rear_half_angle = math.radians(45.0)

        front_min = math.inf
        left_min = math.inf
        right_min = math.inf
        rear_min = math.inf

        for i, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance <= msg.range_min or distance >= msg.range_max:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            angle = self.normalize_angle(angle)

            if abs(angle) <= front_half_angle:
                front_min = min(front_min, distance)
            if abs(self.normalize_angle(angle - math.pi * 0.5)) <= side_half_angle:
                left_min = min(left_min, distance)
            if abs(self.normalize_angle(angle + math.pi * 0.5)) <= side_half_angle:
                right_min = min(right_min, distance)
            if abs(abs(angle) - math.pi) <= rear_half_angle:
                rear_min = min(rear_min, distance)

        return front_min, left_min, right_min, rear_min

    def is_hard_safety_stop(self) -> bool:
        return (
            self.front_min < self.emergency_stop_dist
            or self.left_min < self.side_emergency_dist
            or self.right_min < self.side_emergency_dist
            or self.rear_min < self.rear_emergency_dist
        )

    def start_recovery(self, reason: str):
        if self.recovery_active:
            return

        self.recovery_active = True
        self.recovery_started_time = self.get_clock().now()
        self.recovery_turn_direction = self.choose_clearer_turn_direction()
        self.path_world = []
        self.force_replan = True
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.get_logger().warn(
            'Recovery started: '
            f'{reason}; front={self.front_min:.2f}, left={self.left_min:.2f}, '
            f'right={self.right_min:.2f}, rear={self.rear_min:.2f}',
            throttle_duration_sec=0.5,
        )

    def compute_recovery_command(self) -> Optional[Twist]:
        elapsed = self.elapsed_seconds(
            self.get_clock().now(),
            self.recovery_started_time,
        )

        stop_phase = self.recovery_stop_duration
        back_phase = self.recovery_stop_duration + self.recovery_back_duration

        if elapsed >= back_phase and self.has_recovery_clearance():
            self.recovery_active = False
            self.force_replan = True
            self.get_logger().warn('Recovery finished; replanning.', throttle_duration_sec=0.5)
            return None

        cmd = Twist()

        if elapsed < stop_phase:
            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0
            return cmd

        self.recovery_turn_direction = self.choose_clearer_turn_direction()
        if self.rear_min < self.rear_emergency_dist:
            # Do not reverse into a rear obstacle. Rotate toward the clearer side
            # until the rear sector opens or the map planner can take over.
            cmd.linear.x = 0.0
            cmd.angular.z = self.recovery_turn_direction * self.recovery_turn_speed
        else:
            cmd.linear.x = -self.recovery_back_speed
            cmd.angular.z = self.recovery_turn_direction * self.recovery_turn_speed

        self.last_cmd_v = cmd.linear.x
        self.last_cmd_w = cmd.angular.z
        return cmd

    def has_recovery_clearance(self) -> bool:
        front_clear = (
            not math.isfinite(self.front_min)
            or self.front_min >= self.recovery_clear_front_dist
        )
        left_clear = (
            not math.isfinite(self.left_min)
            or self.left_min >= self.recovery_clear_side_dist
        )
        right_clear = (
            not math.isfinite(self.right_min)
            or self.right_min >= self.recovery_clear_side_dist
        )
        rear_clear = (
            not math.isfinite(self.rear_min)
            or self.rear_min >= self.rear_emergency_dist
        )
        return front_clear and left_clear and right_clear and rear_clear

    def choose_clearer_turn_direction(self) -> float:
        # Positive angular.z turns left.
        if self.left_min >= self.right_min:
            return 1.0
        return -1.0

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------

    def publish_path(self, path: List[WorldPoint], publisher):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for x, y in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        publisher.publish(msg)

    def publish_goal_marker(self):
        if self.goal_xy is None:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'fast_astar_smooth_nav'
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_xy[0]
        marker.pose.position.y = self.goal_xy[1]
        marker.pose.position.z = 0.04
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.28
        marker.scale.y = 0.28
        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.05
        marker.color.b = 0.05
        marker.color.a = 0.9
        self.goal_marker_pub.publish(marker)

    def publish_local_target_marker(self, target: WorldPoint):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'fast_astar_smooth_nav'
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target[0]
        marker.pose.position.y = target[1]
        marker.pose.position.z = 0.08
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.16
        marker.color.r = 1.0
        marker.color.g = 0.75
        marker.color.b = 0.05
        marker.color.a = 0.95
        self.local_target_marker_pub.publish(marker)

    def publish_window_marker(self, robot_xy: WorldPoint):
        if self.start_pose is None:
            return

        corners_start_frame = [
            (-self.window_back_margin, -self.window_half_width),
            (self.target_distance + self.window_goal_margin, -self.window_half_width),
            (self.target_distance + self.window_goal_margin, self.window_half_width),
            (-self.window_back_margin, self.window_half_width),
            (-self.window_back_margin, -self.window_half_width),
        ]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'fast_astar_smooth_nav'
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.6

        for forward, lateral in corners_start_frame:
            x, y = self.from_start_frame(forward, lateral)
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.03
            marker.points.append(p)

        self.window_marker_pub.publish(marker)

    # ---------------------------------------------------------------------
    # Stop / shutdown
    # ---------------------------------------------------------------------

    def stop_robot(self):
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotNavHardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
