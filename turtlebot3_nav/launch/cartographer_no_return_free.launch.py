import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    map_profile = LaunchConfiguration('map', default='house') # 'house' or 'world'

    turtlebot3_cartographer_prefix = get_package_share_directory('turtlebot3_cartographer')
    cartographer_config_dir = LaunchConfiguration(
        'cartographer_config_dir',
        default=os.path.join(turtlebot3_cartographer_prefix, 'config'),
    )
    configuration_basename = PythonExpression([
        "'turtlebot3_lds_2d_house_no_return_free.lua' if '",
        map_profile,
        "' == 'house' else 'turtlebot3_lds_2d_no_return_free.lua'",
    ])

    resolution = LaunchConfiguration('resolution', default='0.05')
    publish_period_sec = LaunchConfiguration('publish_period_sec', default='1.0')
    synthetic_no_return_range = LaunchConfiguration(
        'synthetic_no_return_range',
        default='3.49',
    )
    cartographer_max_range = LaunchConfiguration(
        'cartographer_max_range',
        default='3.20',
    )
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)
    synthetic_no_return_range_param = ParameterValue(
        synthetic_no_return_range,
        value_type=float,
    )
    cartographer_max_range_param = ParameterValue(
        cartographer_max_range,
        value_type=float,
    )

    rviz_config_dir = os.path.join(
        turtlebot3_cartographer_prefix,
        'rviz',
        'tb3_cartographer.rviz',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock if true',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz with the TurtleBot3 Cartographer config',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='world',
            description='Cartographer profile: world or house',
        ),
        DeclareLaunchArgument(
            'cartographer_config_dir',
            default_value=cartographer_config_dir,
            description='Full path to Cartographer lua config directory',
        ),
        DeclareLaunchArgument(
            'resolution',
            default_value=resolution,
            description='Published occupancy grid resolution',
        ),
        DeclareLaunchArgument(
            'publish_period_sec',
            default_value=publish_period_sec,
            description='Occupancy grid publish period',
        ),
        DeclareLaunchArgument(
            'synthetic_no_return_range',
            default_value=synthetic_no_return_range,
            description='Finite scan range used for no-return rays',
        ),
        DeclareLaunchArgument(
            'cartographer_max_range',
            default_value=cartographer_max_range,
            description='Must match TRAJECTORY_BUILDER_2D.max_range in lua',
        ),

        Node(
            package='turtlebot3_nav',
            executable='scan_no_return_filter',
            name='scan_no_return_filter',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time_param,
                'input_scan_topic': '/scan',
                'output_scan_topic': '/scan_cartographer',
                'synthetic_no_return_range': synthetic_no_return_range_param,
                'cartographer_max_range': cartographer_max_range_param,
                'replace_inf': True,
                'replace_nan': True,
                'replace_range_max': True,
                'range_max_epsilon': 0.01,
                'report_period': 5.0,
            }],
        ),

        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time_param}],
            arguments=[
                '-configuration_directory', cartographer_config_dir,
                '-configuration_basename', configuration_basename,
            ],
            remappings=[
                ('scan', 'scan_cartographer'),
            ],
        ),

        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time_param}],
            arguments=[
                '-resolution', resolution,
                '-publish_period_sec', publish_period_sec,
            ],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': use_sim_time_param}],
            condition=IfCondition(use_rviz),
            output='screen',
        ),
    ])
