import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanNoReturnFilter(Node):
    """Convert no-return LaserScan rays into Cartographer miss rays.

    Cartographer can insert free space along a ray when the range value is
    finite but longer than TRAJECTORY_BUILDER_2D.max_range. Gazebo and many
    laser drivers often publish no-return rays as inf, nan, or range_max.
    Those values may be dropped before Cartographer's 2D range data inserter
    sees them, so the open sector stays unknown.

    This node keeps the original /scan untouched for safety logic and publishes
    a Cartographer-only scan topic where no-return rays are represented by a
    finite synthetic distance. The Cartographer lua file must set max_range
    below this synthetic distance so these rays are treated as misses, not hits.
    """

    def __init__(self):
        super().__init__('scan_no_return_filter')

        self.declare_parameter('input_scan_topic', '/scan')
        self.declare_parameter('output_scan_topic', '/scan_cartographer')
        self.declare_parameter('synthetic_no_return_range', 3.49)
        self.declare_parameter('cartographer_max_range', 3.20)
        self.declare_parameter('replace_inf', True)
        self.declare_parameter('replace_nan', True)
        self.declare_parameter('replace_range_max', True)
        self.declare_parameter('range_max_epsilon', 0.01)
        self.declare_parameter('report_period', 5.0)

        self.input_scan_topic = (
            self.get_parameter('input_scan_topic').get_parameter_value().string_value
        )
        self.output_scan_topic = (
            self.get_parameter('output_scan_topic').get_parameter_value().string_value
        )
        self.synthetic_no_return_range = (
            self.get_parameter('synthetic_no_return_range').get_parameter_value().double_value
        )
        self.cartographer_max_range = (
            self.get_parameter('cartographer_max_range').get_parameter_value().double_value
        )
        self.replace_inf = (
            self.get_parameter('replace_inf').get_parameter_value().bool_value
        )
        self.replace_nan = (
            self.get_parameter('replace_nan').get_parameter_value().bool_value
        )
        self.replace_range_max = (
            self.get_parameter('replace_range_max').get_parameter_value().bool_value
        )
        self.range_max_epsilon = (
            self.get_parameter('range_max_epsilon').get_parameter_value().double_value
        )
        self.report_period = (
            self.get_parameter('report_period').get_parameter_value().double_value
        )

        if self.synthetic_no_return_range <= self.cartographer_max_range:
            self.get_logger().warn(
                'synthetic_no_return_range should be larger than '
                'cartographer_max_range. Otherwise no-return rays can become '
                'fake obstacle hits instead of free-space miss rays.'
            )

        self.publisher = self.create_publisher(
            LaserScan,
            self.output_scan_topic,
            qos_profile_sensor_data,
        )
        self.subscription = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.last_report_wall_time = 0.0
        self.get_logger().info(
            'Filtering no-return scan rays: '
            f'{self.input_scan_topic} -> {self.output_scan_topic}, '
            f'synthetic={self.synthetic_no_return_range:.2f} m, '
            f'cartographer_max={self.cartographer_max_range:.2f} m'
        )

    def scan_callback(self, scan_msg):
        filtered_msg = LaserScan()
        filtered_msg.header = scan_msg.header
        filtered_msg.angle_min = scan_msg.angle_min
        filtered_msg.angle_max = scan_msg.angle_max
        filtered_msg.angle_increment = scan_msg.angle_increment
        filtered_msg.time_increment = scan_msg.time_increment
        filtered_msg.scan_time = scan_msg.scan_time
        filtered_msg.range_min = scan_msg.range_min
        filtered_msg.range_max = max(scan_msg.range_max, self.synthetic_no_return_range)
        filtered_msg.intensities = list(scan_msg.intensities)

        filtered_ranges = []
        inf_count = 0
        nan_count = 0
        max_count = 0

        no_return_threshold = scan_msg.range_max - self.range_max_epsilon

        for raw_range in scan_msg.ranges:
            filtered_range = raw_range

            if math.isnan(raw_range):
                if self.replace_nan:
                    filtered_range = self.synthetic_no_return_range
                    nan_count += 1
            elif math.isinf(raw_range):
                if self.replace_inf and raw_range > 0.0:
                    filtered_range = self.synthetic_no_return_range
                    inf_count += 1
            elif (
                self.replace_range_max
                and scan_msg.range_max > scan_msg.range_min
                and raw_range >= no_return_threshold
            ):
                # A range exactly at range_max usually means "nothing returned"
                # in Gazebo ray sensors. Keep near/far real hits unchanged below
                # this threshold.
                filtered_range = self.synthetic_no_return_range
                max_count += 1

            filtered_ranges.append(filtered_range)

        filtered_msg.ranges = filtered_ranges
        self.publisher.publish(filtered_msg)
        self.report_replacements(inf_count, nan_count, max_count, len(filtered_ranges))

    def report_replacements(self, inf_count, nan_count, max_count, total_count):
        now = time.monotonic()
        if now - self.last_report_wall_time < self.report_period:
            return

        self.last_report_wall_time = now
        replaced_count = inf_count + nan_count + max_count
        self.get_logger().info(
            'No-return scan filter replaced '
            f'{replaced_count}/{total_count} rays '
            f'(inf={inf_count}, nan={nan_count}, range_max={max_count}).'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ScanNoReturnFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
