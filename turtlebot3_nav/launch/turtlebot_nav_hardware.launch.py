from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_nav',
            executable='turtlebot_nav_hardware',
            name='turtlebot_nav_hardware',
            output='screen',
            parameters=[{
                'goal_x': 5.5,
                'goal_y': 0.0,
            }],
        ),
    ])
