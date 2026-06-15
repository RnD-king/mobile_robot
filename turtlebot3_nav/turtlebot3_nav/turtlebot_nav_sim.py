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


class TurtlebotNavSim(Node):
    """A simple online-map navigation node for TurtleBot3 Gazebo simulation.

    Cartographer is responsible for SLAM and publishes /map plus map->odom TF.
    This node uses only the currently known OccupancyGrid, repeatedly plans with
    A*, follows the generated waypoints, and uses /scan only as an emergency
    stop layer.
    """

    def __init__(self):
        super().__init__('turtlebot_nav_sim')

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
            OccupancyGrid, '/map', self.map_callback, map_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.local_target_pub = self.create_publisher(PoseStamped, '/local_target', 10)
        self.local_target_marker_pub = self.create_publisher(
            Marker, '/local_target_marker', 10
        )
        self.goal_marker_pub = self.create_publisher(
            Marker, '/goal_marker', 10
        )
        self.active_subgoal_pub = self.create_publisher(
            PoseStamped, '/active_subgoal', 10
        )
        self.active_subgoal_marker_pub = self.create_publisher(
            Marker, '/active_subgoal_marker', 10
        )
        self.scan_free_space_marker_pub = self.create_publisher(
            Marker, '/scan_free_space_marker', 10
        )
        self.nav_augmented_map_pub = self.create_publisher(
            OccupancyGrid, '/nav_augmented_map', 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.costmap: Optional[List[float]] = None
        self.blocked: Optional[List[bool]] = None
        self.map_dirty = False
        self.latest_scan: Optional[LaserScan] = None
        self.scan_free_cells: Set[GridCell] = set()
        self.scan_blocked_cells: Set[GridCell] = set()
        self.scan_free_cell_stamps: Dict[GridCell, int] = {}
        self.recent_scan_free_cell_set: Set[GridCell] = set()

        self.goal_xy: Optional[WorldPoint] = None
        self.active_subgoal_xy: Optional[WorldPoint] = None
        self.active_subgoal_kind = 'none'
        self.active_subgoal_score = math.inf
        self.active_subgoal_set_time = self.get_clock().now()
        self.active_subgoal_fail_count = 0
        self.blacklisted_subgoals: List[WorldPoint] = []
        self.visited_reachable_subgoals: List[WorldPoint] = []
        self.path_world: List[WorldPoint] = []
        self.local_target_xy: Optional[WorldPoint] = None
        self.last_plan_time = self.get_clock().now()
        self.force_replan = True
        self.last_progress_xy: Optional[WorldPoint] = None
        self.last_progress_time = self.get_clock().now()

        self.front_min = math.inf
        self.left_min = math.inf
        self.right_min = math.inf
        self.rear_min = math.inf
        self.emergency_active = False
        self.last_emergency_log = self.get_clock().now()
        self.recovery_mode = 'idle'
        self.recovery_end_time = self.get_clock().now()
        self.recovery_turn_direction = 1.0

        if self.use_param_goal:
            self.goal_xy = (self.goal_x, self.goal_y)
            self.get_logger().info(
                f'Using parameter goal: x={self.goal_x:.3f}, y={self.goal_y:.3f}'
            )
        else:
            self.get_logger().info(
                'Waiting for RViz goal on /move_base_simple/goal.'
            )

        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop
        )

    def _declare_parameters(self):
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('fallback_base_frame', 'base_link')
        self.declare_parameter('use_param_goal', False)
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.35)
        self.declare_parameter('yaw_gain', 1.9)
        self.declare_parameter('lookahead_distance', 0.50)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('replan_period', 1.0)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('unknown_cost', 6.0)
        self.declare_parameter('allow_unknown', True)
        self.declare_parameter('inflation_radius', 0.11)
        self.declare_parameter('soft_inflation_radius', 0.24)
        self.declare_parameter('soft_inflation_cost', 8.0)
        self.declare_parameter('path_smoothing_enabled', True)
        self.declare_parameter('smoothing_clearance_radius', 0.16)
        self.declare_parameter('emergency_stop_dist', 0.14)
        self.declare_parameter('side_emergency_dist', 0.12)
        self.declare_parameter('rear_emergency_dist', 0.10)
        self.declare_parameter('front_angle_deg', 30.0)
        self.declare_parameter('goal_block_search_radius', 0.45)
        self.declare_parameter('map_edge_margin', 0.20)
        self.declare_parameter('frontier_candidate_limit', 50)
        self.declare_parameter('frontier_min_cluster_size', 4)
        self.declare_parameter('frontier_reached_tolerance', 0.35)
        self.declare_parameter('frontier_info_radius', 0.35)
        self.declare_parameter('frontier_goal_weight', 1.2)
        self.declare_parameter('frontier_robot_weight', 0.35)
        self.declare_parameter('frontier_info_weight', 0.12)
        self.declare_parameter('frontier_switch_margin', 0.8)
        self.declare_parameter('subgoal_commit_time', 4.0)
        self.declare_parameter('subgoal_early_release_distance', 0.55)
        self.declare_parameter('subgoal_blacklist_radius', 0.35)
        self.declare_parameter('reachable_switch_margin', 0.35)
        self.declare_parameter('reachable_visit_radius', 0.45)
        self.declare_parameter('reachable_visit_penalty', 2.0)
        self.declare_parameter('reachable_history_size', 20)
        self.declare_parameter('reachable_path_cost_weight', 0.25)
        self.declare_parameter('optimistic_planning_margin', 3.0)
        self.declare_parameter('optimistic_unknown_cost', 5.0)
        self.declare_parameter('optimistic_outside_map_cost', 8.0)
        self.declare_parameter('optimistic_max_expansions', 80000)
        self.declare_parameter('subgoal_direction_switch_angle', 0.55)
        self.declare_parameter('subgoal_large_switch_improvement', 1.20)
        self.declare_parameter('subgoal_same_direction_tolerance', 0.05)
        self.declare_parameter('local_target_switch_angle', 0.45)
        self.declare_parameter('local_target_reached_distance', 0.16)
        self.declare_parameter('local_target_path_tolerance', 0.30)
        self.declare_parameter('local_target_large_switch_improvement', 0.35)
        self.declare_parameter('rotation_debug_period', 2.0)
        self.declare_parameter('min_linear_speed_scale', 0.40)
        self.declare_parameter('curvature_speed_gain', 0.45)
        self.declare_parameter('obstacle_slow_dist', 0.45)
        self.declare_parameter('side_slow_dist', 0.24)
        self.declare_parameter('obstacle_repulsion_dist', 0.34)
        self.declare_parameter('obstacle_repulsion_gain', 0.60)
        self.declare_parameter('side_balance_gain', 0.25)
        self.declare_parameter('front_turn_gain', 0.55)
        self.declare_parameter('rotate_in_place_yaw', 1.10)
        self.declare_parameter('stuck_timeout', 4.0)
        self.declare_parameter('stuck_progress_distance', 0.08)
        self.declare_parameter('recovery_enabled', True)
        self.declare_parameter('recovery_turn_speed', 0.95)
        self.declare_parameter('recovery_turn_duration', 1.1)
        self.declare_parameter('recovery_forward_speed', 0.08)
        self.declare_parameter('recovery_forward_duration', 0.8)
        self.declare_parameter('use_scan_free_space_layer', True)
        self.declare_parameter('scan_free_space_max_range', 3.0)
        self.declare_parameter('scan_free_space_cost', 1.6)
        self.declare_parameter('scan_free_space_ray_step', 0.05)
        self.declare_parameter('scan_obstacle_endpoint_margin', 0.18)
        self.declare_parameter('publish_scan_free_space_marker', True)
        self.declare_parameter('publish_nav_augmented_map', True)
        self.declare_parameter('scan_free_space_ttl', 5.0)

    def _load_parameters(self):
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.fallback_base_frame = self.get_parameter('fallback_base_frame').value
        self.use_param_goal = bool(self.get_parameter('use_param_goal').value)
        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.yaw_gain = float(self.get_parameter('yaw_gain').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.control_frequency = float(self.get_parameter('control_frequency').value)
        self.replan_period = float(self.get_parameter('replan_period').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.unknown_cost = float(self.get_parameter('unknown_cost').value)
        self.allow_unknown = bool(self.get_parameter('allow_unknown').value)
        self.inflation_radius = float(self.get_parameter('inflation_radius').value)
        self.soft_inflation_radius = float(
            self.get_parameter('soft_inflation_radius').value
        )
        self.soft_inflation_cost = float(
            self.get_parameter('soft_inflation_cost').value
        )
        self.path_smoothing_enabled = bool(
            self.get_parameter('path_smoothing_enabled').value
        )
        self.smoothing_clearance_radius = float(
            self.get_parameter('smoothing_clearance_radius').value
        )
        self.emergency_stop_dist = float(
            self.get_parameter('emergency_stop_dist').value
        )
        self.side_emergency_dist = float(
            self.get_parameter('side_emergency_dist').value
        )
        self.rear_emergency_dist = float(
            self.get_parameter('rear_emergency_dist').value
        )
        self.front_angle_deg = float(self.get_parameter('front_angle_deg').value)
        self.goal_block_search_radius = float(
            self.get_parameter('goal_block_search_radius').value
        )
        self.map_edge_margin = float(self.get_parameter('map_edge_margin').value)
        self.frontier_candidate_limit = int(
            self.get_parameter('frontier_candidate_limit').value
        )
        self.frontier_min_cluster_size = int(
            self.get_parameter('frontier_min_cluster_size').value
        )
        self.frontier_reached_tolerance = float(
            self.get_parameter('frontier_reached_tolerance').value
        )
        self.frontier_info_radius = float(
            self.get_parameter('frontier_info_radius').value
        )
        self.frontier_goal_weight = float(
            self.get_parameter('frontier_goal_weight').value
        )
        self.frontier_robot_weight = float(
            self.get_parameter('frontier_robot_weight').value
        )
        self.frontier_info_weight = float(
            self.get_parameter('frontier_info_weight').value
        )
        self.frontier_switch_margin = float(
            self.get_parameter('frontier_switch_margin').value
        )
        self.subgoal_commit_time = float(
            self.get_parameter('subgoal_commit_time').value
        )
        self.subgoal_early_release_distance = float(
            self.get_parameter('subgoal_early_release_distance').value
        )
        self.subgoal_blacklist_radius = float(
            self.get_parameter('subgoal_blacklist_radius').value
        )
        self.reachable_switch_margin = float(
            self.get_parameter('reachable_switch_margin').value
        )
        self.reachable_visit_radius = float(
            self.get_parameter('reachable_visit_radius').value
        )
        self.reachable_visit_penalty = float(
            self.get_parameter('reachable_visit_penalty').value
        )
        self.reachable_history_size = int(
            self.get_parameter('reachable_history_size').value
        )
        self.reachable_path_cost_weight = float(
            self.get_parameter('reachable_path_cost_weight').value
        )
        self.optimistic_planning_margin = float(
            self.get_parameter('optimistic_planning_margin').value
        )
        self.optimistic_unknown_cost = float(
            self.get_parameter('optimistic_unknown_cost').value
        )
        self.optimistic_outside_map_cost = float(
            self.get_parameter('optimistic_outside_map_cost').value
        )
        self.optimistic_max_expansions = int(
            self.get_parameter('optimistic_max_expansions').value
        )
        self.subgoal_direction_switch_angle = float(
            self.get_parameter('subgoal_direction_switch_angle').value
        )
        self.subgoal_large_switch_improvement = float(
            self.get_parameter('subgoal_large_switch_improvement').value
        )
        self.subgoal_same_direction_tolerance = float(
            self.get_parameter('subgoal_same_direction_tolerance').value
        )
        self.local_target_switch_angle = float(
            self.get_parameter('local_target_switch_angle').value
        )
        self.local_target_reached_distance = float(
            self.get_parameter('local_target_reached_distance').value
        )
        self.local_target_path_tolerance = float(
            self.get_parameter('local_target_path_tolerance').value
        )
        self.local_target_large_switch_improvement = float(
            self.get_parameter('local_target_large_switch_improvement').value
        )
        self.rotation_debug_period = float(
            self.get_parameter('rotation_debug_period').value
        )
        self.min_linear_speed_scale = float(
            self.get_parameter('min_linear_speed_scale').value
        )
        self.curvature_speed_gain = float(
            self.get_parameter('curvature_speed_gain').value
        )
        self.obstacle_slow_dist = float(
            self.get_parameter('obstacle_slow_dist').value
        )
        self.side_slow_dist = float(
            self.get_parameter('side_slow_dist').value
        )
        self.obstacle_repulsion_dist = float(
            self.get_parameter('obstacle_repulsion_dist').value
        )
        self.obstacle_repulsion_gain = float(
            self.get_parameter('obstacle_repulsion_gain').value
        )
        self.side_balance_gain = float(
            self.get_parameter('side_balance_gain').value
        )
        self.front_turn_gain = float(
            self.get_parameter('front_turn_gain').value
        )
        self.rotate_in_place_yaw = float(
            self.get_parameter('rotate_in_place_yaw').value
        )
        self.stuck_timeout = float(self.get_parameter('stuck_timeout').value)
        self.stuck_progress_distance = float(
            self.get_parameter('stuck_progress_distance').value
        )
        self.recovery_enabled = bool(self.get_parameter('recovery_enabled').value)
        self.recovery_turn_speed = float(
            self.get_parameter('recovery_turn_speed').value
        )
        self.recovery_turn_duration = float(
            self.get_parameter('recovery_turn_duration').value
        )
        self.recovery_forward_speed = float(
            self.get_parameter('recovery_forward_speed').value
        )
        self.recovery_forward_duration = float(
            self.get_parameter('recovery_forward_duration').value
        )
        self.use_scan_free_space_layer = bool(
            self.get_parameter('use_scan_free_space_layer').value
        )
        self.scan_free_space_max_range = float(
            self.get_parameter('scan_free_space_max_range').value
        )
        self.scan_free_space_cost = float(
            self.get_parameter('scan_free_space_cost').value
        )
        self.scan_free_space_ray_step = float(
            self.get_parameter('scan_free_space_ray_step').value
        )
        self.scan_obstacle_endpoint_margin = float(
            self.get_parameter('scan_obstacle_endpoint_margin').value
        )
        self.publish_scan_free_space_marker_enabled = bool(
            self.get_parameter('publish_scan_free_space_marker').value
        )
        self.publish_nav_augmented_map_enabled = bool(
            self.get_parameter('publish_nav_augmented_map').value
        )
        self.scan_free_space_ttl = float(
            self.get_parameter('scan_free_space_ttl').value
        )

    # ----------------------------- ROS callbacks ----------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.costmap, self.blocked = self.build_inflated_costmap(msg)
        self.map_dirty = True

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.front_min, self.left_min, self.right_min, self.rear_min = (
            self.compute_sector_min_distances(msg)
        )

    def goal_callback(self, msg: PoseStamped):
        frame_id = msg.header.frame_id.strip()
        if frame_id and frame_id != self.map_frame:
            self.get_logger().warn(
                f'Goal frame is {frame_id}, expected {self.map_frame}. '
                'Using the numeric pose as-is.'
            )

        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.clear_active_subgoal()
        self.blacklisted_subgoals = []
        self.visited_reachable_subgoals = []
        self.clear_local_target()
        self.last_progress_xy = None
        self.path_world = []
        self.map_dirty = True
        self.force_replan = True
        self.publish_goal_marker()
        self.delete_active_subgoal_marker()
        self.get_logger().info(
            f'New goal received: x={self.goal_xy[0]:.3f}, y={self.goal_xy[1]:.3f}'
        )

    # ------------------------------- Main loop ------------------------------

    def control_loop(self):
        if self.map_msg is None or self.costmap is None or self.blocked is None:
            self.stop_robot()
            return

        pose = self.lookup_robot_pose()
        if pose is None:
            self.stop_robot()
            return

        robot_x, robot_y, robot_yaw = pose
        self.update_scan_free_space_layer(robot_x, robot_y, robot_yaw)
        self.update_persistent_scan_free_space()
        self.publish_scan_free_space_marker()
        self.publish_nav_augmented_map()

        if self.goal_xy is None:
            self.clear_local_target()
            self.stop_robot()
            return
        self.publish_goal_marker()

        if self.distance((robot_x, robot_y), self.goal_xy) <= self.goal_tolerance:
            self.stop_robot()
            if self.path_world:
                self.get_logger().info('Goal reached.')
            self.path_world = []
            self.clear_local_target()
            self.publish_path([])
            self.clear_active_subgoal()
            return

        if self.recovery_enabled and self.recovery_mode != 'idle':
            self.clear_local_target()
            self.cmd_pub.publish(self.compute_recovery_command())
            return

        if self.is_safety_stop():
            self.log_safety_stop()
            if self.recovery_enabled:
                self.clear_local_target()
                self.get_logger().warn(
                    'Recovery requested by safety stop.',
                    throttle_duration_sec=1.0,
                )
                self.cmd_pub.publish(self.compute_recovery_command())
            else:
                self.stop_robot()
            return

        self.recovery_mode = 'idle'

        if self.update_progress_monitor((robot_x, robot_y), robot_yaw):
            self.get_logger().warn(
                'Stuck detected: clearing path and starting recovery.',
                throttle_duration_sec=2.0,
            )
            self.abandon_active_subgoal('robot did not make enough progress')
            self.path_world = []
            self.map_dirty = True
            self.force_replan = True
            if self.recovery_enabled:
                self.cmd_pub.publish(self.compute_recovery_command())
            else:
                self.stop_robot()
            return

        if self.need_replan(robot_x, robot_y):
            if not self.plan_path(robot_x, robot_y):
                self.stop_robot()
                return

        cmd = self.compute_velocity_command(robot_x, robot_y, robot_yaw)
        self.cmd_pub.publish(cmd)

    # ---------------------------- TF and geometry ---------------------------

    def lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
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
    def elapsed_seconds(now, past) -> float:
        return max(0.0, (now.nanoseconds - past.nanoseconds) / 1.0e9)

    @staticmethod
    def distance(a: WorldPoint, b: WorldPoint) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # -------------------------- OccupancyGrid utils -------------------------

    def world_to_grid(self, x: float, y: float) -> Optional[GridCell]:
        gx, gy = self.world_to_grid_unbounded(x, y)
        if not self.is_inside_grid(gx, gy):
            return None
        return gx, gy

    def world_to_grid_unbounded(self, x: float, y: float) -> GridCell:
        info = self.map_msg.info
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        gx = int(math.floor((x - origin_x) / info.resolution))
        gy = int(math.floor((y - origin_y) / info.resolution))
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

    def update_scan_free_space_layer(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ):
        self.scan_free_cells = set()
        self.scan_blocked_cells = set()

        if (
            not self.use_scan_free_space_layer
            or self.latest_scan is None
            or self.map_msg is None
            or self.blocked is None
        ):
            return

        scan = self.latest_scan
        resolution = self.map_msg.info.resolution
        ray_step = max(resolution * 0.5, self.scan_free_space_ray_step)
        max_range = self.scan_free_space_max_range
        if math.isfinite(scan.range_max) and scan.range_max > scan.range_min:
            max_range = min(max_range, scan.range_max)
        if max_range <= scan.range_min:
            return

        self.scan_free_cells.add(self.world_to_grid_unbounded(robot_x, robot_y))

        inflation_cells = max(
            1, int(math.ceil(self.inflation_radius / resolution))
        )
        obstacle_offsets = self.make_inflation_offsets(inflation_cells)

        for i, raw_distance in enumerate(scan.ranges):
            if math.isnan(raw_distance):
                continue

            has_hit = False
            if math.isfinite(raw_distance):
                if raw_distance <= scan.range_min:
                    continue
                if raw_distance < scan.range_max:
                    has_hit = raw_distance <= max_range
                    usable_distance = min(raw_distance, max_range)
                else:
                    usable_distance = max_range
            else:
                usable_distance = max_range

            free_distance = usable_distance
            if has_hit:
                free_distance = max(
                    0.0, usable_distance - self.scan_obstacle_endpoint_margin
                )

            angle = robot_yaw + scan.angle_min + i * scan.angle_increment
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            distance = 0.0
            while distance <= free_distance:
                wx = robot_x + distance * cos_a
                wy = robot_y + distance * sin_a
                cell = self.world_to_grid_unbounded(wx, wy)
                if self.is_inside_grid(*cell) and self.blocked[self.grid_index(cell)]:
                    break
                self.scan_free_cells.add(cell)
                distance += ray_step

            if has_hit:
                hit_x = robot_x + usable_distance * cos_a
                hit_y = robot_y + usable_distance * sin_a
                hit_cell = self.world_to_grid_unbounded(hit_x, hit_y)
                for dx, dy in obstacle_offsets:
                    self.scan_blocked_cells.add((hit_cell[0] + dx, hit_cell[1] + dy))

        self.scan_free_cells.difference_update(self.scan_blocked_cells)

    def update_persistent_scan_free_space(self):
        if not self.use_scan_free_space_layer:
            self.scan_free_cell_stamps = {}
            self.recent_scan_free_cell_set = set()
            return

        now_ns = self.get_clock().now().nanoseconds
        ttl_ns = int(max(0.0, self.scan_free_space_ttl) * 1.0e9)

        for cell in self.scan_free_cells:
            if cell not in self.scan_blocked_cells and self.is_scan_cell_safe(cell):
                self.scan_free_cell_stamps[cell] = now_ns

        for cell in self.scan_blocked_cells:
            self.scan_free_cell_stamps.pop(cell, None)

        if ttl_ns <= 0:
            self.scan_free_cell_stamps = {}
            self.recent_scan_free_cell_set = set()
            return

        stale_cells = [
            cell for cell, stamp_ns in self.scan_free_cell_stamps.items()
            if now_ns - stamp_ns > ttl_ns
        ]
        for cell in stale_cells:
            self.scan_free_cell_stamps.pop(cell, None)

        self.recent_scan_free_cell_set = (
            set(self.scan_free_cell_stamps.keys()) | self.scan_free_cells
        )

    def recent_scan_free_cells(self) -> Set[GridCell]:
        return self.recent_scan_free_cell_set

    def is_scan_cell_safe(self, cell: GridCell) -> bool:
        if cell in self.scan_blocked_cells:
            return False
        if self.is_inside_grid(*cell):
            return self.blocked is not None and not self.blocked[self.grid_index(cell)]
        return True

    def is_plannable_cell(self, cell: GridCell) -> bool:
        if cell in self.scan_blocked_cells:
            return False
        if self.is_inside_grid(*cell):
            return self.blocked is not None and not self.blocked[self.grid_index(cell)]
        return cell in self.recent_scan_free_cells()

    def is_reachable_free_cell(self, cell: GridCell) -> bool:
        return self.is_free_space_cell(cell)

    def is_free_space_cell(self, cell: GridCell) -> bool:
        if cell in self.scan_blocked_cells:
            return False
        if cell in self.recent_scan_free_cells():
            return self.is_scan_cell_safe(cell)
        if not self.is_inside_grid(*cell):
            return False
        if self.blocked is None or self.blocked[self.grid_index(cell)]:
            return False
        return self.is_known_free_cell(cell)

    def cell_traversal_cost(self, cell: GridCell) -> float:
        if cell in self.scan_blocked_cells:
            return math.inf
        if self.is_inside_grid(*cell):
            cost = self.costmap[self.grid_index(cell)]
            if cell in self.recent_scan_free_cells():
                return max(cost, self.scan_free_space_cost)
            return cost
        if cell in self.recent_scan_free_cells():
            return self.scan_free_space_cost
        return math.inf

    def is_optimistic_plannable_cell(self, cell: GridCell) -> bool:
        if cell in self.scan_blocked_cells:
            return False
        if not self.is_inside_grid(*cell):
            return True

        idx = self.grid_index(cell)
        value = self.map_msg.data[idx]
        if value >= self.occupied_threshold:
            return False
        if self.blocked is not None and self.blocked[idx]:
            return False
        return True

    def optimistic_cell_traversal_cost(self, cell: GridCell) -> float:
        if cell in self.scan_blocked_cells:
            return math.inf
        if not self.is_inside_grid(*cell):
            if cell in self.recent_scan_free_cells():
                return self.scan_free_space_cost
            return self.optimistic_outside_map_cost

        idx = self.grid_index(cell)
        value = self.map_msg.data[idx]
        if value >= self.occupied_threshold:
            return math.inf
        if self.blocked is not None and self.blocked[idx]:
            return math.inf
        if value < 0:
            return self.optimistic_unknown_cost

        cost = self.costmap[idx]
        if cell in self.recent_scan_free_cells():
            return max(cost, self.scan_free_space_cost)
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
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = candidate
            if best_cell is not None:
                return best_cell

        return None

    def nearest_free_space_cell(
        self, cell: GridCell, max_radius_m: float
    ) -> Optional[GridCell]:
        if self.is_free_space_cell(cell):
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
                    if not self.is_free_space_cell(candidate):
                        continue
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = candidate
            if best_cell is not None:
                return best_cell

        return None

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

        # Convert OccupancyGrid values into an internal cost layer.
        # - occupied cells are blocked
        # - unknown cells are either blocked or expensive, depending on params
        # - free cells get base traversal cost 1.0
        for idx, value in enumerate(msg.data):
            if value >= self.occupied_threshold:
                blocked[idx] = True
                gx = idx % width
                gy = idx // width
                obstacle_cells.append((gx, gy))
            elif value < 0:
                if self.allow_unknown:
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
                if 0 <= nx < width and 0 <= ny < height:
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
        offsets = []
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dx, dy))
        return offsets

    def nearest_unblocked_cell(self, cell: GridCell, max_radius_m: float) -> Optional[GridCell]:
        return self.nearest_plannable_cell(cell, max_radius_m)

    def world_point_to_unblocked_cell(
        self, point: WorldPoint, search_radius_m: float
    ) -> Optional[GridCell]:
        cell = self.world_to_grid_unbounded(point[0], point[1])
        return self.nearest_plannable_cell(cell, search_radius_m)

    def is_known_free_cell(self, cell: GridCell) -> bool:
        if not self.is_inside_grid(*cell):
            return False
        value = self.map_msg.data[self.grid_index(cell)]
        return 0 <= value < self.occupied_threshold

    def is_blacklisted_subgoal(self, point: WorldPoint) -> bool:
        for blocked_point in self.blacklisted_subgoals:
            if self.distance(point, blocked_point) <= self.subgoal_blacklist_radius:
                return True
        return False

    def clear_active_subgoal(self):
        had_subgoal = self.active_subgoal_xy is not None
        self.active_subgoal_xy = None
        self.active_subgoal_kind = 'none'
        self.active_subgoal_score = math.inf
        self.active_subgoal_fail_count = 0
        if had_subgoal:
            self.delete_active_subgoal_marker()

    def abandon_active_subgoal(self, reason: str):
        if self.active_subgoal_xy is None:
            return

        if self.active_subgoal_kind in ('reachable', 'optimistic'):
            self.remember_reachable_subgoal(self.active_subgoal_xy)

        if self.active_subgoal_kind != 'goal':
            self.blacklisted_subgoals.append(self.active_subgoal_xy)
            self.get_logger().warn(
                f'Abandoning {self.active_subgoal_kind} subgoal: {reason}.',
                throttle_duration_sec=1.0,
            )

        self.clear_active_subgoal()
        self.force_replan = True

    def set_active_subgoal(self, point: WorldPoint, kind: str, score: float = math.inf):
        changed = (
            self.active_subgoal_xy is None
            or self.active_subgoal_kind != kind
            or self.distance(self.active_subgoal_xy, point) > 0.05
        )
        self.active_subgoal_xy = point
        self.active_subgoal_kind = kind
        self.active_subgoal_score = score
        self.active_subgoal_fail_count = 0
        self.active_subgoal_set_time = self.get_clock().now()
        if changed:
            self.get_logger().info(
                f'Active subgoal: {kind} x={point[0]:.3f}, y={point[1]:.3f}'
            )

    def active_subgoal_elapsed(self) -> float:
        return self.elapsed_seconds(
            self.get_clock().now(), self.active_subgoal_set_time
        )

    def active_subgoal_committed(self, robot_xy: Optional[WorldPoint] = None) -> bool:
        if self.active_subgoal_xy is None:
            return False
        if self.active_subgoal_kind == 'goal':
            return False
        if (
            robot_xy is not None
            and self.distance(robot_xy, self.active_subgoal_xy)
            <= self.subgoal_early_release_distance
        ):
            return False
        return self.active_subgoal_elapsed() < self.subgoal_commit_time

    def clear_active_subgoal_if_reached(self, robot_xy: WorldPoint):
        if self.active_subgoal_xy is None:
            return
        tolerance = self.goal_tolerance
        if self.active_subgoal_kind != 'goal':
            tolerance = self.frontier_reached_tolerance
        if self.distance(robot_xy, self.active_subgoal_xy) <= tolerance:
            self.get_logger().info(
                f'Reached active {self.active_subgoal_kind} subgoal.',
                throttle_duration_sec=1.0,
            )
            if self.active_subgoal_kind in ('reachable', 'optimistic'):
                self.remember_reachable_subgoal(self.active_subgoal_xy)
            self.clear_active_subgoal()

    def remember_reachable_subgoal(self, point: WorldPoint):
        self.visited_reachable_subgoals.append(point)
        if self.reachable_history_size <= 0:
            self.visited_reachable_subgoals = []
            return
        self.visited_reachable_subgoals = self.visited_reachable_subgoals[
            -self.reachable_history_size:
        ]

    def reachable_visit_penalty_for(self, point: WorldPoint) -> float:
        if self.reachable_visit_radius <= 0.0:
            return 0.0
        penalty = 0.0
        for visited in self.visited_reachable_subgoals:
            dist = self.distance(point, visited)
            if dist >= self.reachable_visit_radius:
                continue
            penalty += self.reachable_visit_penalty * (
                1.0 - dist / self.reachable_visit_radius
            )
        return penalty

    def is_near_visited_reachable_subgoal(self, point: WorldPoint) -> bool:
        if self.reachable_visit_radius <= 0.0:
            return False
        return any(
            self.distance(point, visited) < self.reachable_visit_radius
            for visited in self.visited_reachable_subgoals
        )

    def should_switch_subgoal(
        self,
        robot_xy: WorldPoint,
        candidate_point: WorldPoint,
        candidate_kind: str,
        candidate_goal_distance: float,
    ) -> bool:
        if self.active_subgoal_xy is None:
            return True
        if self.active_subgoal_kind == 'goal':
            return True

        current_goal_distance = self.distance(self.goal_xy, self.active_subgoal_xy)
        improvement = current_goal_distance - candidate_goal_distance
        direction_change = self.subgoal_direction_change(
            robot_xy,
            self.active_subgoal_xy,
            candidate_point,
        )

        if direction_change <= self.subgoal_direction_switch_angle:
            return improvement >= -self.subgoal_same_direction_tolerance

        if improvement >= self.subgoal_large_switch_improvement:
            self.get_logger().info(
                f'Accepting large {candidate_kind} subgoal switch: '
                f'improvement={improvement:.2f} m, '
                f'direction_change={direction_change:.2f} rad.',
                throttle_duration_sec=2.0,
            )
            return True

        self.get_logger().info(
            f'Keeping current {self.active_subgoal_kind} subgoal; '
            f'{candidate_kind} switch is too abrupt '
            f'(improvement={improvement:.2f} m, '
            f'direction_change={direction_change:.2f} rad).',
            throttle_duration_sec=2.0,
        )
        return False

    def subgoal_direction_change(
        self,
        robot_xy: WorldPoint,
        current_point: WorldPoint,
        candidate_point: WorldPoint,
    ) -> float:
        current_dist = self.distance(robot_xy, current_point)
        candidate_dist = self.distance(robot_xy, candidate_point)
        if current_dist < 1.0e-3 or candidate_dist < 1.0e-3:
            return 0.0

        current_yaw = math.atan2(
            current_point[1] - robot_xy[1],
            current_point[0] - robot_xy[0],
        )
        candidate_yaw = math.atan2(
            candidate_point[1] - robot_xy[1],
            candidate_point[0] - robot_xy[0],
        )
        return abs(self.normalize_angle(candidate_yaw - current_yaw))

    # ---------------------------------- A* ----------------------------------

    def plan_path(self, robot_x: float, robot_y: float) -> bool:
        self.last_plan_time = self.get_clock().now()
        self.force_replan = False
        robot_xy = (robot_x, robot_y)
        self.clear_active_subgoal_if_reached(robot_xy)

        start = self.world_to_grid_unbounded(robot_x, robot_y)
        start = self.nearest_free_space_cell(start, self.goal_block_search_radius)
        if start is None:
            start = self.nearest_plannable_cell(
                self.world_to_grid_unbounded(robot_x, robot_y),
                self.goal_block_search_radius,
            )
        if start is None:
            self.get_logger().warn(
                'No plannable start cell found.',
                throttle_duration_sec=1.0,
            )
            return False

        planned = self.plan_to_best_target(start, robot_xy)
        grid_path, target_kind = planned

        self.map_dirty = False

        if not grid_path:
            self.get_logger().warn(
                'A* failed to find a path or useful frontier.',
                throttle_duration_sec=1.0,
            )
            if self.path_world and not self.is_path_blocked(robot_x, robot_y):
                self.get_logger().warn(
                    'Keeping previous path after temporary planning failure.',
                    throttle_duration_sec=2.0,
                )
                return True
            self.path_world = []
            self.clear_local_target()
            self.publish_path([])
            return False

        grid_path = self.smooth_grid_path(
            grid_path,
            free_space_only=target_kind in ('goal', 'reachable', 'optimistic'),
        )
        self.path_world = [self.grid_to_world(cell) for cell in grid_path]
        self.refresh_local_target_after_replan(robot_xy)
        self.publish_path(self.path_world)
        if self.active_subgoal_xy is not None:
            self.publish_active_subgoal()
        self.get_logger().info(
            f'Planned {target_kind} path with {len(self.path_world)} waypoints.',
            throttle_duration_sec=1.0,
        )
        return True

    def plan_to_best_target(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> Tuple[List[GridCell], str]:
        if self.active_subgoal_committed(robot_xy):
            active_path = self.try_plan_to_active_subgoal(start)
            if active_path:
                return active_path, self.active_subgoal_kind
            self.drop_failed_active_subgoal()

        if self.goal_is_in_free_space():
            goal_path = self.try_plan_to_final_goal(start)
            if goal_path:
                self.set_active_subgoal(self.goal_xy, 'goal', 0.0)
                return goal_path, 'goal'
        else:
            optimistic_path = self.try_plan_to_optimistic_subgoal(start, robot_xy)
            if optimistic_path:
                return optimistic_path, self.active_subgoal_kind

            if self.active_subgoal_xy is not None:
                active_path = self.try_plan_to_active_subgoal(start)
                if active_path:
                    self.get_logger().info(
                        'Keeping current subgoal after temporary optimistic failure.',
                        throttle_duration_sec=2.0,
                    )
                    return active_path, self.active_subgoal_kind

            reachable_path = self.try_plan_to_closest_reachable_subgoal(
                start, robot_xy
            )
            if reachable_path:
                return reachable_path, self.active_subgoal_kind

        if self.active_subgoal_xy is not None:
            active_path = self.try_plan_to_active_subgoal(start)
            if active_path:
                return active_path, self.active_subgoal_kind
            self.drop_failed_active_subgoal()

        frontier_path = self.try_plan_to_frontier_subgoal(start, robot_xy)
        if frontier_path:
            return frontier_path, 'frontier'

        map_edge_path = self.try_plan_to_map_edge_subgoal(start, robot_xy)
        if map_edge_path:
            return map_edge_path, 'map-edge'

        return [], 'none'

    def drop_failed_active_subgoal(self):
        self.active_subgoal_fail_count += 1
        if (
            self.active_subgoal_xy is not None
            and self.active_subgoal_kind != 'goal'
        ):
            if self.active_subgoal_kind in ('reachable', 'optimistic'):
                self.remember_reachable_subgoal(self.active_subgoal_xy)
            self.blacklisted_subgoals.append(self.active_subgoal_xy)
        self.clear_active_subgoal()

    def goal_is_in_free_space(self) -> bool:
        goal = self.world_to_grid_unbounded(self.goal_xy[0], self.goal_xy[1])
        return self.is_free_space_cell(goal)

    def try_plan_to_final_goal(self, start: GridCell) -> List[GridCell]:
        goal = self.world_to_grid_unbounded(self.goal_xy[0], self.goal_xy[1])
        if not self.is_free_space_cell(goal):
            return []
        return self.astar(start, goal, free_space_only=True)

    def try_plan_to_active_subgoal(self, start: GridCell) -> List[GridCell]:
        if self.active_subgoal_xy is None:
            return []

        active_cell = self.world_to_grid_unbounded(
            self.active_subgoal_xy[0], self.active_subgoal_xy[1]
        )

        if self.active_subgoal_kind == 'optimistic':
            active_cell = self.nearest_plannable_cell(
                active_cell, self.goal_block_search_radius
            )
        else:
            active_cell = self.nearest_free_space_cell(
                active_cell, self.goal_block_search_radius
            )
        if active_cell is None:
            return []

        path = self.astar(start, active_cell, free_space_only=True)
        if path:
            return path

        if self.active_subgoal_kind == 'optimistic':
            bounds = self.optimistic_planning_bounds(start, active_cell)
            path = self.astar(
                start,
                active_cell,
                optimistic=True,
                bounds=bounds,
                max_expansions=self.optimistic_max_expansions,
            )
            if path:
                self.get_logger().info(
                    'Keeping optimistic subgoal through unknown/grey cells.',
                    throttle_duration_sec=2.0,
                )
                return path

        return []

    def try_plan_to_optimistic_subgoal(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[GridCell]:
        goal = self.world_to_grid_unbounded(self.goal_xy[0], self.goal_xy[1])
        bounds = self.optimistic_planning_bounds(start, goal)
        optimistic_path = self.astar(
            start,
            goal,
            optimistic=True,
            bounds=bounds,
            max_expansions=self.optimistic_max_expansions,
        )
        if len(optimistic_path) < 2:
            return self.try_plan_to_optimistic_frontier_subgoal(start, robot_xy)

        min_robot_distance = max(0.20, min(self.lookahead_distance, 0.35))
        chosen_idx = None
        for idx, cell in enumerate(optimistic_path):
            if not self.is_reachable_free_cell(cell):
                break
            if idx == 0:
                continue
            point = self.grid_to_world(cell)
            if self.distance(robot_xy, point) < min_robot_distance:
                continue
            if self.is_blacklisted_subgoal(point):
                continue
            chosen_idx = idx

        if chosen_idx is None:
            return self.try_plan_to_optimistic_frontier_subgoal(start, robot_xy)

        subgoal_cell = optimistic_path[chosen_idx]
        subgoal_point = self.grid_to_world(subgoal_cell)
        goal_distance = self.distance(self.goal_xy, subgoal_point)

        if not self.should_switch_subgoal(
            robot_xy,
            subgoal_point,
            'optimistic',
            goal_distance,
        ):
            active_path = self.try_plan_to_active_subgoal(start)
            if active_path:
                return active_path

        if (
            self.active_subgoal_kind == 'optimistic'
            and self.active_subgoal_xy is not None
            and self.distance(self.active_subgoal_xy, subgoal_point) > 0.05
        ):
            self.remember_reachable_subgoal(self.active_subgoal_xy)

        self.set_active_subgoal(subgoal_point, 'optimistic', goal_distance)
        self.get_logger().info(
            'Goal is outside known free-space; using optimistic path prefix.',
            throttle_duration_sec=2.0,
        )
        return optimistic_path[:chosen_idx + 1]

    def try_plan_to_optimistic_frontier_subgoal(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[GridCell]:
        reachable = self.reachable_known_free_cells(start)
        if not reachable:
            return []

        min_robot_distance = max(0.30, self.lookahead_distance)
        candidates: List[Tuple[float, float, GridCell]] = []
        for cell in reachable:
            point = self.grid_to_world(cell)
            if self.distance(robot_xy, point) < min_robot_distance:
                continue
            if self.is_blacklisted_subgoal(point):
                continue
            if not self.is_frontier_cell(cell):
                continue

            goal_distance = self.distance(self.goal_xy, point)
            visit_penalty = self.reachable_visit_penalty_for(point)
            path_hint = self.distance(robot_xy, point)
            score = (
                goal_distance
                + visit_penalty
                + self.reachable_path_cost_weight * path_hint
            )
            candidates.append((score, goal_distance, cell))

        candidates.sort(key=lambda item: item[0])

        best_path: List[GridCell] = []
        best_point: Optional[WorldPoint] = None
        best_goal_distance = math.inf
        best_score = math.inf

        for candidate_score, goal_distance, candidate in candidates[:100]:
            path = self.astar(start, candidate, free_space_only=True)
            if not path or len(path) < 2:
                continue
            path_cost = self.grid_path_cost(path) * self.map_msg.info.resolution
            score = candidate_score + self.reachable_path_cost_weight * path_cost
            point = self.grid_to_world(candidate)
            if score < best_score:
                best_score = score
                best_goal_distance = goal_distance
                best_point = point
                best_path = path

        if not best_path or best_point is None:
            return []

        if not self.should_switch_subgoal(
            robot_xy,
            best_point,
            'optimistic-frontier',
            best_goal_distance,
        ):
            active_path = self.try_plan_to_active_subgoal(start)
            if active_path:
                return active_path

        if (
            self.active_subgoal_kind in ('optimistic', 'reachable')
            and self.active_subgoal_xy is not None
            and self.distance(self.active_subgoal_xy, best_point) > 0.05
        ):
            self.remember_reachable_subgoal(self.active_subgoal_xy)

        self.set_active_subgoal(best_point, 'optimistic', best_goal_distance)
        self.get_logger().info(
            'Optimistic path to final goal failed; using goal-directed frontier.',
            throttle_duration_sec=2.0,
        )
        return best_path

    def optimistic_planning_bounds(
        self, start: GridCell, goal: GridCell
    ) -> Tuple[int, int, int, int]:
        margin_cells = max(
            1,
            int(math.ceil(
                self.optimistic_planning_margin / self.map_msg.info.resolution
            )),
        )
        width = self.map_msg.info.width
        height = self.map_msg.info.height
        min_x = min(0, start[0], goal[0]) - margin_cells
        max_x = max(width - 1, start[0], goal[0]) + margin_cells
        min_y = min(0, start[1], goal[1]) - margin_cells
        max_y = max(height - 1, start[1], goal[1]) + margin_cells
        return min_x, max_x, min_y, max_y

    @staticmethod
    def is_cell_in_bounds(cell: GridCell, bounds: Tuple[int, int, int, int]) -> bool:
        min_x, max_x, min_y, max_y = bounds
        return min_x <= cell[0] <= max_x and min_y <= cell[1] <= max_y

    def try_plan_to_closest_reachable_subgoal(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[GridCell]:
        reachable = self.reachable_known_free_cells(start)
        if not reachable:
            return []

        min_robot_distance = max(0.20, min(self.lookahead_distance, 0.35))
        fresh_candidates: List[Tuple[float, float, GridCell]] = []
        visited_candidates: List[Tuple[float, float, GridCell]] = []
        for cell in reachable:
            point = self.grid_to_world(cell)
            if self.distance(robot_xy, point) < min_robot_distance:
                continue
            if self.is_blacklisted_subgoal(point):
                continue
            goal_distance = self.distance(self.goal_xy, point)
            visit_penalty = self.reachable_visit_penalty_for(point)
            candidate = (goal_distance + visit_penalty, goal_distance, cell)
            if self.is_near_visited_reachable_subgoal(point):
                visited_candidates.append(candidate)
            else:
                fresh_candidates.append(candidate)

        candidates = fresh_candidates if fresh_candidates else visited_candidates
        candidates.sort(key=lambda item: item[0])

        best_path: List[GridCell] = []
        best_point: Optional[WorldPoint] = None
        best_goal_distance = math.inf
        best_score = math.inf

        for candidate_score, goal_distance, candidate in candidates[:80]:
            path = self.astar(start, candidate, free_space_only=True)
            if not path or len(path) < 2:
                continue
            path_cost = self.grid_path_cost(path)
            score = (
                candidate_score
                + self.reachable_path_cost_weight
                * path_cost
                * self.map_msg.info.resolution
            )
            point = self.grid_to_world(candidate)
            if score < best_score:
                best_score = score
                best_goal_distance = goal_distance
                best_point = point
                best_path = path

        if not best_path or best_point is None:
            return []

        if not self.should_switch_subgoal(
            robot_xy,
            best_point,
            'reachable',
            best_goal_distance,
        ):
            active_path = self.try_plan_to_active_subgoal(start)
            if active_path:
                return active_path

        if (
            self.active_subgoal_kind == 'reachable'
            and self.active_subgoal_xy is not None
            and self.distance(self.active_subgoal_xy, best_point) > 0.05
        ):
            self.remember_reachable_subgoal(self.active_subgoal_xy)

        self.set_active_subgoal(best_point, 'reachable', best_goal_distance)
        self.get_logger().info(
            'Goal is outside known free-space; using closest reachable subgoal.',
            throttle_duration_sec=2.0,
        )
        return best_path

    def grid_path_cost(self, path: List[GridCell]) -> float:
        if len(path) < 2:
            return 0.0

        cost = 0.0
        prev = path[0]
        for cell in path[1:]:
            step = math.hypot(cell[0] - prev[0], cell[1] - prev[1])
            cost += step * max(1.0, self.cell_traversal_cost(cell))
            prev = cell
        return cost

    def try_plan_to_map_edge_subgoal(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[GridCell]:
        if self.world_to_grid(self.goal_xy[0], self.goal_xy[1]) is not None:
            return []
        subgoal = self.goal_direction_subgoal(robot_xy, self.goal_xy)
        if subgoal is None:
            return []
        subgoal = self.nearest_unblocked_cell(subgoal, self.goal_block_search_radius)
        if subgoal is None:
            return []
        point = self.grid_to_world(subgoal)
        if self.is_blacklisted_subgoal(point):
            return []
        path = self.astar(start, subgoal)
        if path:
            self.set_active_subgoal(
                point,
                'map-edge',
                self.distance(point, self.goal_xy),
            )
            self.get_logger().info(
                'Final goal is outside current map; using map-edge subgoal.',
                throttle_duration_sec=2.0,
            )
        return path

    def try_plan_to_frontier_subgoal(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[GridCell]:
        candidates = self.frontier_candidates(start, robot_xy)
        current_score = self.active_subgoal_score
        if self.active_subgoal_xy is not None:
            current_score = self.active_subgoal_score

        for score, candidate in candidates:
            point = self.grid_to_world(candidate)
            if self.is_blacklisted_subgoal(point):
                continue
            if (
                self.active_subgoal_xy is not None
                and self.active_subgoal_kind == 'frontier'
                and score + self.frontier_switch_margin >= current_score
            ):
                active_cell = self.world_point_to_unblocked_cell(
                    self.active_subgoal_xy, self.goal_block_search_radius
                )
                if active_cell is not None:
                    active_path = self.astar(start, active_cell)
                    if active_path:
                        return active_path
            path = self.astar(start, candidate)
            if path:
                self.set_active_subgoal(point, 'frontier', score)
                self.get_logger().info(
                    'Using reachable goal-directed frontier subgoal.',
                    throttle_duration_sec=2.0,
                )
                return path
            self.blacklisted_subgoals.append(point)
        return []

    def goal_direction_subgoal(
        self, robot_xy: WorldPoint, goal_xy: WorldPoint
    ) -> Optional[GridCell]:
        bounds = self.map_world_bounds(self.map_edge_margin)
        if bounds is None:
            return None

        min_x, max_x, min_y, max_y = bounds
        rx, ry = robot_xy
        gx, gy = goal_xy
        dx = gx - rx
        dy = gy - ry
        if math.hypot(dx, dy) < 1.0e-6:
            return None

        candidates = []
        if dx > 0.0:
            candidates.append((max_x - rx) / dx)
        elif dx < 0.0:
            candidates.append((min_x - rx) / dx)

        if dy > 0.0:
            candidates.append((max_y - ry) / dy)
        elif dy < 0.0:
            candidates.append((min_y - ry) / dy)

        valid_t = [t for t in candidates if t > 0.0]
        if not valid_t:
            return None

        t = max(0.0, min(1.0, min(valid_t)))
        target_x = rx + dx * t
        target_y = ry + dy * t
        target_x = max(min_x, min(max_x, target_x))
        target_y = max(min_y, min(max_y, target_y))
        return self.world_to_grid(target_x, target_y)

    def map_world_bounds(self, margin: float) -> Optional[Tuple[float, float, float, float]]:
        if self.map_msg is None:
            return None

        info = self.map_msg.info
        min_x = info.origin.position.x + margin
        min_y = info.origin.position.y + margin
        max_x = info.origin.position.x + info.width * info.resolution - margin
        max_y = info.origin.position.y + info.height * info.resolution - margin
        if min_x >= max_x or min_y >= max_y:
            return None
        return min_x, max_x, min_y, max_y

    def frontier_candidates(
        self, start: GridCell, robot_xy: WorldPoint
    ) -> List[Tuple[float, GridCell]]:
        reachable = self.reachable_known_free_cells(start)
        frontier_cells = {
            cell for cell in reachable
            if self.is_frontier_cell(cell)
            and self.distance(robot_xy, self.grid_to_world(cell))
            >= max(0.35, self.lookahead_distance)
        }
        clusters = self.cluster_frontiers(frontier_cells)

        scored: List[Tuple[float, GridCell]] = []
        for cluster in clusters:
            if len(cluster) < self.frontier_min_cluster_size:
                continue
            representative = self.frontier_cluster_representative(cluster)
            if representative is None:
                continue
            world = self.grid_to_world(representative)
            robot_dist = self.distance(robot_xy, world)
            goal_dist = self.distance(self.goal_xy, world)
            info_gain = self.frontier_information_gain(representative)
            switch_penalty = 0.0
            if (
                self.active_subgoal_xy is not None
                and self.active_subgoal_kind == 'frontier'
            ):
                switch_penalty = 0.2 * self.distance(self.active_subgoal_xy, world)
            score = (
                self.frontier_goal_weight * goal_dist
                + self.frontier_robot_weight * robot_dist
                - self.frontier_info_weight * info_gain
                + switch_penalty
            )
            scored.append((score, representative))

        scored.sort(key=lambda item: item[0])
        return scored[:self.frontier_candidate_limit]

    def reachable_known_free_cells(self, start: GridCell) -> Set[GridCell]:
        reachable: Set[GridCell] = set()
        queue = deque()
        if self.is_reachable_free_cell(start):
            reachable.add(start)
            queue.append(start)

        while queue:
            cell = queue.popleft()
            for neighbor in self.get_free_neighbors(cell):
                if neighbor in reachable:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)

        return reachable

    def get_free_neighbors(self, cell: GridCell) -> List[GridCell]:
        result = []
        gx, gy = cell
        for dx, dy in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            neighbor = (gx + dx, gy + dy)
            if not self.is_reachable_free_cell(neighbor):
                continue
            if dx != 0 and dy != 0:
                side_a = (gx + dx, gy)
                side_b = (gx, gy + dy)
                if (
                    not self.is_reachable_free_cell(side_a)
                    or not self.is_reachable_free_cell(side_b)
                ):
                    continue
            result.append(neighbor)
        return result

    def cluster_frontiers(self, frontier_cells: Set[GridCell]) -> List[List[GridCell]]:
        clusters: List[List[GridCell]] = []
        remaining = set(frontier_cells)
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            queue = deque([seed])
            while queue:
                gx, gy = queue.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (gx + dx, gy + dy)
                        if neighbor not in remaining:
                            continue
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cluster.append(neighbor)
            clusters.append(cluster)
        return clusters

    def frontier_cluster_representative(
        self, cluster: List[GridCell]
    ) -> Optional[GridCell]:
        if not cluster:
            return None
        cx = sum(cell[0] for cell in cluster) / len(cluster)
        cy = sum(cell[1] for cell in cluster) / len(cluster)
        return min(cluster, key=lambda cell: math.hypot(cell[0] - cx, cell[1] - cy))

    def frontier_information_gain(self, cell: GridCell) -> int:
        radius = max(1, int(math.ceil(self.frontier_info_radius / self.map_msg.info.resolution)))
        gx, gy = cell
        gain = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                nx = gx + dx
                ny = gy + dy
                if not self.is_inside_grid(nx, ny):
                    gain += 1
                    continue
                if self.map_msg.data[ny * self.map_msg.info.width + nx] < 0:
                    gain += 1
        return gain

    def is_frontier_cell(self, cell: GridCell) -> bool:
        gx, gy = cell
        if not self.is_reachable_free_cell(cell):
            return False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = gx + dx
                ny = gy + dy
                neighbor = (nx, ny)
                if not self.is_inside_grid(nx, ny):
                    return True
                if (
                    self.map_msg.data[ny * self.map_msg.info.width + nx] < 0
                    and neighbor not in self.recent_scan_free_cells()
                ):
                    return True
        return False

    def astar(
        self,
        start: GridCell,
        goal: GridCell,
        free_space_only: bool = False,
        optimistic: bool = False,
        bounds: Optional[Tuple[int, int, int, int]] = None,
        max_expansions: int = 0,
    ) -> List[GridCell]:
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[GridCell, GridCell] = {}
        g_score: Dict[GridCell, float] = {start: 0.0}
        closed = set()
        expansions = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self.reconstruct_path(came_from, current)
            closed.add(current)
            expansions += 1
            if max_expansions > 0 and expansions > max_expansions:
                return []

            for neighbor, step_cost in self.get_neighbors(
                current,
                free_space_only=free_space_only,
                optimistic=optimistic,
                bounds=bounds,
            ):
                if neighbor in closed:
                    continue

                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return []

    def get_neighbors(
        self,
        cell: GridCell,
        free_space_only: bool = False,
        optimistic: bool = False,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Tuple[GridCell, float]]:
        result = []
        gx, gy = cell
        directions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        for dx, dy, move_cost in directions:
            neighbor = (gx + dx, gy + dy)
            if not self.is_cell_allowed_for_planning(
                neighbor,
                free_space_only=free_space_only,
                optimistic=optimistic,
                bounds=bounds,
            ):
                continue

            # Avoid squeezing diagonally through two blocked corner cells.
            if dx != 0 and dy != 0:
                side_a = (gx + dx, gy)
                side_b = (gx, gy + dy)
                side_a_ok = self.is_cell_allowed_for_planning(
                    side_a,
                    free_space_only=free_space_only,
                    optimistic=optimistic,
                    bounds=bounds,
                )
                side_b_ok = self.is_cell_allowed_for_planning(
                    side_b,
                    free_space_only=free_space_only,
                    optimistic=optimistic,
                    bounds=bounds,
                )
                if not side_a_ok or not side_b_ok:
                    continue

            if optimistic:
                traversal_cost = self.optimistic_cell_traversal_cost(neighbor)
            else:
                traversal_cost = self.cell_traversal_cost(neighbor)
            if not math.isfinite(traversal_cost):
                continue
            result.append((neighbor, move_cost * traversal_cost))

        return result

    def is_cell_allowed_for_planning(
        self,
        cell: GridCell,
        free_space_only: bool = False,
        optimistic: bool = False,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        if bounds is not None and not self.is_cell_in_bounds(cell, bounds):
            return False
        if free_space_only:
            return self.is_reachable_free_cell(cell)
        if optimistic:
            return self.is_optimistic_plannable_cell(cell)
        return self.is_plannable_cell(cell)

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

    def smooth_grid_path(
        self, path: List[GridCell], free_space_only: bool = False
    ) -> List[GridCell]:
        if not self.path_smoothing_enabled or len(path) <= 2:
            return path

        smoothed = [path[0]]
        clearance_cells = max(
            0,
            int(math.ceil(
                self.smoothing_clearance_radius / self.map_msg.info.resolution
            )),
        )
        anchor_idx = 0
        while anchor_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > anchor_idx + 1:
                if self.has_line_of_sight(
                    path[anchor_idx],
                    path[next_idx],
                    clearance_cells,
                    free_space_only=free_space_only,
                ):
                    break
                next_idx -= 1
            smoothed.append(path[next_idx])
            anchor_idx = next_idx

        return smoothed

    def has_line_of_sight(
        self,
        start: GridCell,
        end: GridCell,
        clearance_cells: int = 0,
        free_space_only: bool = False,
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
            if not self.has_planning_clearance(
                (x, y),
                clearance_cells,
                free_space_only=free_space_only,
            ):
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

    def has_planning_clearance(
        self,
        cell: GridCell,
        clearance_cells: int,
        free_space_only: bool = False,
    ) -> bool:
        if clearance_cells <= 0:
            if free_space_only:
                return self.is_reachable_free_cell(cell)
            return self.is_plannable_cell(cell)

        cx, cy = cell
        radius_sq = clearance_cells * clearance_cells
        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):
                if dx * dx + dy * dy > radius_sq:
                    continue
                candidate = (cx + dx, cy + dy)
                if free_space_only:
                    candidate_ok = self.is_reachable_free_cell(candidate)
                else:
                    candidate_ok = self.is_plannable_cell(candidate)
                if not candidate_ok:
                    return False
        return True

    # ----------------------------- Path tracking ----------------------------

    def need_replan(self, robot_x: float, robot_y: float) -> bool:
        now = self.get_clock().now()

        if self.force_replan:
            return True

        if not self.path_world:
            return self.elapsed_seconds(now, self.last_plan_time) > self.replan_period

        if self.elapsed_seconds(now, self.last_plan_time) > self.replan_period:
            return True

        if self.map_dirty and self.is_path_blocked(robot_x, robot_y):
            return True

        return False

    def is_path_blocked(self, robot_x: float, robot_y: float) -> bool:
        if self.blocked is None:
            return True

        checked = 0
        for point in self.path_world:
            if self.distance((robot_x, robot_y), point) < self.lookahead_distance * 0.5:
                continue
            cell = self.world_to_grid_unbounded(point[0], point[1])
            if not self.is_plannable_cell(cell):
                return True
            checked += 1
            if checked >= 20:
                break
        return False

    def update_progress_monitor(self, robot_xy: WorldPoint, robot_yaw: float) -> bool:
        now = self.get_clock().now()

        if not self.path_world:
            self.last_progress_xy = robot_xy
            self.last_progress_time = now
            return False

        target = self.select_lookahead_point(robot_xy[0], robot_xy[1])
        if target is None:
            self.last_progress_xy = robot_xy
            self.last_progress_time = now
            return False

        heading_distance = self.distance(robot_xy, target)
        if heading_distance < self.goal_tolerance:
            self.last_progress_xy = robot_xy
            self.last_progress_time = now
            return False

        if self.last_progress_xy is None:
            self.last_progress_xy = robot_xy
            self.last_progress_time = now
            return False

        target_yaw = math.atan2(target[1] - robot_xy[1], target[0] - robot_xy[0])
        yaw_error = abs(self.normalize_angle(target_yaw - robot_yaw))
        if yaw_error > self.rotate_in_place_yaw * 0.8:
            if self.distance(robot_xy, self.last_progress_xy) >= self.stuck_progress_distance:
                self.last_progress_xy = robot_xy
                self.last_progress_time = now
                return False
            rotation_timeout = max(self.stuck_timeout, 5.0)
            return self.elapsed_seconds(now, self.last_progress_time) > rotation_timeout

        if self.distance(robot_xy, self.last_progress_xy) >= self.stuck_progress_distance:
            self.last_progress_xy = robot_xy
            self.last_progress_time = now
            return False

        return self.elapsed_seconds(now, self.last_progress_time) > self.stuck_timeout

    def compute_velocity_command(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ) -> Twist:
        cmd = Twist()
        target = self.select_lookahead_point(robot_x, robot_y)
        if target is None:
            return cmd
        self.publish_local_target(target)

        target_yaw = math.atan2(target[1] - robot_y, target[0] - robot_x)
        yaw_error = self.normalize_angle(target_yaw - robot_yaw)
        lookahead = max(0.05, self.distance((robot_x, robot_y), target))
        curvature = abs(2.0 * math.sin(yaw_error) / lookahead)
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.yaw_gain * yaw_error),
        )
        repulsion_angular = self.obstacle_repulsion_angular()
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular + repulsion_angular),
        )

        if abs(yaw_error) > self.rotate_in_place_yaw:
            linear = 0.0
            if self.rotation_debug_period > 0.0:
                self.get_logger().info(
                    'Rotate-in-place control: '
                    f'robot=({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.2f}), '
                    f'target=({target[0]:.2f}, {target[1]:.2f}), '
                    f'target_yaw={target_yaw:.2f}, yaw_error={yaw_error:.2f}, '
                    f'angular={angular:.2f}',
                    throttle_duration_sec=self.rotation_debug_period,
                )
        else:
            yaw_scale = max(
                self.min_linear_speed_scale,
                1.0 - abs(yaw_error) / self.rotate_in_place_yaw,
            )
            curvature_scale = 1.0 / (1.0 + self.curvature_speed_gain * curvature)
            obstacle_scale = self.front_obstacle_speed_scale()
            side_scale = self.side_obstacle_speed_scale()
            cost_scale = self.local_cost_speed_scale(target)
            linear = (
                self.linear_speed
                * yaw_scale
                * curvature_scale
                * obstacle_scale
                * side_scale
                * cost_scale
            )

        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd

    def front_obstacle_speed_scale(self) -> float:
        if not math.isfinite(self.front_min):
            return 1.0
        if self.front_min <= self.emergency_stop_dist:
            return 0.0
        if self.front_min >= self.obstacle_slow_dist:
            return 1.0
        span = max(0.01, self.obstacle_slow_dist - self.emergency_stop_dist)
        ratio = (self.front_min - self.emergency_stop_dist) / span
        return max(self.min_linear_speed_scale, min(1.0, ratio))

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
        return max(self.min_linear_speed_scale, min(1.0, ratio))

    def obstacle_repulsion_angular(self) -> float:
        angular = 0.0

        if math.isfinite(self.left_min) and self.left_min < self.obstacle_repulsion_dist:
            ratio = (
                self.obstacle_repulsion_dist - self.left_min
            ) / max(0.01, self.obstacle_repulsion_dist - self.emergency_stop_dist)
            angular -= self.side_balance_gain * max(0.0, min(1.0, ratio))

        if math.isfinite(self.right_min) and self.right_min < self.obstacle_repulsion_dist:
            ratio = (
                self.obstacle_repulsion_dist - self.right_min
            ) / max(0.01, self.obstacle_repulsion_dist - self.emergency_stop_dist)
            angular += self.side_balance_gain * max(0.0, min(1.0, ratio))

        if math.isfinite(self.front_min) and self.front_min < self.obstacle_slow_dist:
            ratio = (
                self.obstacle_slow_dist - self.front_min
            ) / max(0.01, self.obstacle_slow_dist - self.emergency_stop_dist)
            angular += (
                self.choose_recovery_turn_direction()
                * self.front_turn_gain
                * max(0.0, min(1.0, ratio))
            )

        return max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular * self.obstacle_repulsion_gain),
        )

    def local_cost_speed_scale(self, target: WorldPoint) -> float:
        cell = self.world_to_grid_unbounded(target[0], target[1])
        if self.costmap is None:
            return 1.0
        cost = self.cell_traversal_cost(cell)
        if not math.isfinite(cost):
            return self.min_linear_speed_scale
        if cost <= 1.0:
            return 1.0
        return max(self.min_linear_speed_scale, 1.0 / (1.0 + 0.12 * (cost - 1.0)))

    def refresh_local_target_after_replan(self, robot_xy: WorldPoint):
        if self.local_target_xy is None or not self.path_world:
            return

        candidate = self.select_raw_lookahead_point(
            robot_xy[0],
            robot_xy[1],
            prune_path=False,
        )
        if candidate is None:
            self.clear_local_target()
            return

        if not self.is_local_target_usable(robot_xy, self.local_target_xy, candidate):
            self.clear_local_target()
            return

        direction_change = self.subgoal_direction_change(
            robot_xy,
            self.local_target_xy,
            candidate,
        )
        if direction_change > max(0.90, self.local_target_switch_angle * 2.0):
            self.clear_local_target()

    def select_lookahead_point(self, robot_x: float, robot_y: float) -> Optional[WorldPoint]:
        candidate = self.select_raw_lookahead_point(robot_x, robot_y)
        if candidate is None:
            self.clear_local_target()
            return None

        robot_xy = (robot_x, robot_y)
        if not self.is_local_target_usable(robot_xy, self.local_target_xy, candidate):
            self.local_target_xy = candidate
            return self.local_target_xy

        if self.should_switch_local_target(robot_xy, candidate):
            self.local_target_xy = candidate

        return self.local_target_xy

    def select_raw_lookahead_point(
        self, robot_x: float, robot_y: float, prune_path: bool = True
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

        # Drop waypoints behind the nearest point to keep tracking stable.
        path = self.path_world
        if nearest_idx > 0 and prune_path:
            self.path_world = self.path_world[nearest_idx:]
            path = self.path_world
        elif nearest_idx > 0:
            path = self.path_world[nearest_idx:]

        for point in path:
            if self.distance(robot, point) >= self.lookahead_distance:
                return point

        return path[-1]

    def is_local_target_usable(
        self,
        robot_xy: WorldPoint,
        target: Optional[WorldPoint],
        raw_candidate: WorldPoint,
    ) -> bool:
        if target is None or not self.path_world:
            return False
        if self.distance(robot_xy, target) < self.local_target_reached_distance:
            return False

        if self.path_progress_ahead(target, raw_candidate):
            return False

        cell = self.world_to_grid_unbounded(target[0], target[1])
        if not self.is_plannable_cell(cell):
            return False

        return self.distance_to_path(target) <= self.local_target_path_tolerance

    def should_switch_local_target(
        self, robot_xy: WorldPoint, candidate: WorldPoint
    ) -> bool:
        if self.local_target_xy is None:
            return True

        direction_change = self.subgoal_direction_change(
            robot_xy,
            self.local_target_xy,
            candidate,
        )
        if self.path_progress_ahead(self.local_target_xy, candidate):
            return True

        if direction_change <= self.local_target_switch_angle:
            return True

        if self.active_subgoal_xy is not None:
            current_dist = self.distance(self.local_target_xy, self.active_subgoal_xy)
            candidate_dist = self.distance(candidate, self.active_subgoal_xy)
            improvement = current_dist - candidate_dist
            if improvement >= self.local_target_large_switch_improvement:
                return True

        self.get_logger().info(
            f'Keeping local target; candidate jump is too abrupt '
            f'(direction_change={direction_change:.2f} rad).',
            throttle_duration_sec=1.5,
        )
        return False

    def nearest_path_index(self, point: WorldPoint) -> int:
        if not self.path_world:
            return 0

        best_idx = 0
        best_dist = math.inf
        for idx, path_point in enumerate(self.path_world):
            dist = self.distance(point, path_point)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def path_progress_ahead(
        self, current_target: WorldPoint, candidate: WorldPoint
    ) -> bool:
        if not self.path_world:
            return False

        current_idx = self.nearest_path_index(current_target)
        candidate_idx = self.nearest_path_index(candidate)
        if candidate_idx <= current_idx:
            return False

        progress_distance = 0.0
        for idx in range(current_idx, candidate_idx):
            progress_distance += self.distance(
                self.path_world[idx],
                self.path_world[idx + 1],
            )

        return progress_distance >= self.local_target_reached_distance

    def distance_to_path(self, point: WorldPoint) -> float:
        if not self.path_world:
            return math.inf

        best = math.inf
        for path_point in self.path_world:
            best = min(best, self.distance(point, path_point))
        return best

    def publish_path(self, path_world: List[WorldPoint]):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def publish_local_target(self, target: WorldPoint):
        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = target[0]
        pose.pose.position.y = target[1]
        pose.pose.position.z = 0.05
        pose.pose.orientation.w = 1.0
        self.local_target_pub.publish(pose)

        marker = Marker()
        marker.header = pose.header
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.16
        marker.color.r = 1.0
        marker.color.g = 0.75
        marker.color.b = 0.05
        marker.color.a = 0.95
        self.local_target_marker_pub.publish(marker)

    def clear_local_target(self):
        self.local_target_xy = None
        self.delete_local_target_marker()

    def delete_local_target_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 1
        marker.action = Marker.DELETE
        self.local_target_marker_pub.publish(marker)

    def publish_goal_marker(self):
        if self.goal_xy is None:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 10
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
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.9
        self.goal_marker_pub.publish(marker)

    def delete_active_subgoal_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 2
        marker.action = Marker.DELETE
        self.active_subgoal_marker_pub.publish(marker)

    def publish_active_subgoal(self):
        if self.active_subgoal_xy is None:
            return

        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = self.active_subgoal_xy[0]
        pose.pose.position.y = self.active_subgoal_xy[1]
        pose.pose.position.z = 0.08
        pose.pose.orientation.w = 1.0
        self.active_subgoal_pub.publish(pose)

        if self.active_subgoal_kind == 'goal':
            self.delete_active_subgoal_marker()
            return

        marker = Marker()
        marker.header = pose.header
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 2
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.08
        if self.active_subgoal_kind == 'frontier':
            marker.color.r = 0.1
            marker.color.g = 0.45
            marker.color.b = 1.0
        elif self.active_subgoal_kind == 'reachable':
            marker.color.r = 1.0
            marker.color.g = 0.75
            marker.color.b = 0.05
        elif self.active_subgoal_kind == 'optimistic':
            marker.color.r = 0.0
            marker.color.g = 0.9
            marker.color.b = 0.75
        else:
            marker.color.r = 0.8
            marker.color.g = 0.2
            marker.color.b = 1.0
        marker.color.a = 0.9
        self.active_subgoal_marker_pub.publish(marker)

    def publish_scan_free_space_marker(self):
        if not self.publish_scan_free_space_marker_enabled:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'turtlebot_nav_sim'
        marker.id = 3
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        resolution = self.map_msg.info.resolution
        marker.scale.x = resolution
        marker.scale.y = resolution
        marker.scale.z = 0.01
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.05
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.22

        for cell in self.recent_scan_free_cells():
            x, y = self.grid_to_world(cell)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.015
            marker.points.append(point)

        self.scan_free_space_marker_pub.publish(marker)

    def publish_nav_augmented_map(self):
        if not self.publish_nav_augmented_map_enabled or self.map_msg is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info = self.map_msg.info
        data = list(self.map_msg.data)

        for cell in self.recent_scan_free_cells():
            if not self.is_inside_grid(*cell):
                continue
            idx = self.grid_index(cell)
            if self.blocked is not None and self.blocked[idx]:
                continue
            data[idx] = 0

        msg.data = data
        self.nav_augmented_map_pub.publish(msg)

    # ----------------------------- Safety layer -----------------------------

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

    def is_front_danger(self) -> bool:
        return self.front_min < self.emergency_stop_dist

    def is_left_danger(self) -> bool:
        return self.left_min < self.side_emergency_dist

    def is_right_danger(self) -> bool:
        return self.right_min < self.side_emergency_dist

    def is_rear_danger(self) -> bool:
        return self.rear_min < self.rear_emergency_dist

    def is_side_danger(self) -> bool:
        return self.is_left_danger() or self.is_right_danger()

    def is_safety_stop(self) -> bool:
        return (
            self.is_front_danger()
            or self.is_side_danger()
            or self.is_rear_danger()
        )

    def is_recovery_forward_safe(self) -> bool:
        return not (self.is_front_danger() or self.is_side_danger())

    def log_safety_stop(self):
        now = self.get_clock().now()
        if self.elapsed_seconds(now, self.last_emergency_log) > 1.0:
            self.get_logger().warn(
                'Safety stop: '
                f'front={self.front_min:.3f}/{self.emergency_stop_dist:.3f}, '
                f'left={self.left_min:.3f}/{self.side_emergency_dist:.3f}, '
                f'right={self.right_min:.3f}/{self.side_emergency_dist:.3f}, '
                f'rear={self.rear_min:.3f}/{self.rear_emergency_dist:.3f}'
            )
            self.last_emergency_log = now

    def compute_recovery_command(self) -> Twist:
        now = self.get_clock().now()

        if self.recovery_mode == 'forward' and now >= self.recovery_end_time:
            self.recovery_mode = 'idle'
            self.path_world = []
            self.clear_local_target()
            self.map_dirty = True
            self.force_replan = True
            self.last_progress_xy = None
            return Twist()

        if self.recovery_mode == 'forward' and not self.is_recovery_forward_safe():
            self.recovery_turn_direction = self.choose_recovery_turn_direction()
            self.recovery_mode = 'turn'
            self.recovery_end_time = now + Duration(
                seconds=self.recovery_turn_duration
            )
            self.get_logger().warn(
                'Recovery: clearance is still unsafe, turning again.',
                throttle_duration_sec=1.0,
            )

        if self.recovery_mode == 'turn' and now >= self.recovery_end_time:
            if self.is_recovery_forward_safe():
                self.recovery_mode = 'forward'
                self.recovery_end_time = now + Duration(
                    seconds=self.recovery_forward_duration
                )
                self.get_logger().warn(
                    'Recovery: moving after turning toward clear space.',
                    throttle_duration_sec=1.0,
                )
            else:
                self.recovery_turn_direction = self.choose_recovery_turn_direction()
                self.recovery_end_time = now + Duration(
                    seconds=self.recovery_turn_duration
                )
                self.get_logger().warn(
                    'Recovery: not enough clearance, continuing turn.',
                    throttle_duration_sec=1.0,
                )

        if self.recovery_mode == 'idle':
            self.recovery_turn_direction = self.choose_recovery_turn_direction()
            self.recovery_mode = 'turn'
            self.recovery_end_time = now + Duration(
                seconds=self.recovery_turn_duration
            )
            self.get_logger().warn(
                'Recovery: turning toward clearer side before moving.',
                throttle_duration_sec=1.0,
            )

        cmd = Twist()
        if self.recovery_mode == 'turn':
            cmd.angular.z = self.recovery_turn_direction * self.recovery_turn_speed
            self.path_world = []
            self.clear_local_target()
            self.map_dirty = True
        elif self.recovery_mode == 'forward':
            cmd.linear.x = self.recovery_forward_speed
            cmd.angular.z = self.obstacle_repulsion_angular()
            self.path_world = []
            self.clear_local_target()
            self.map_dirty = True
        return cmd

    def choose_recovery_turn_direction(self) -> float:
        # Positive angular velocity turns left. Choose the side with more room.
        if self.is_left_danger() and not self.is_right_danger():
            return -1.0
        if self.is_right_danger() and not self.is_left_danger():
            return 1.0
        if self.left_min >= self.right_min:
            return 1.0
        return -1.0

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotNavSim()
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
