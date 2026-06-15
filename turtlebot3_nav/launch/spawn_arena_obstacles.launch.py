from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare('turtlebot3_nav')

    default_obstacles_file = PathJoinSubstitution([
        package_share,
        'config',
        'arena_obstacles.yaml',
    ])

    default_models_dir = PathJoinSubstitution([
        package_share,
        'models',
    ])

    obstacles_file = LaunchConfiguration('obstacles_file')
    models_dir = LaunchConfiguration('models_dir')

    return LaunchDescription([
        DeclareLaunchArgument(
            'obstacles_file',
            default_value=default_obstacles_file,
            description='Path to arena_obstacles.yaml',
        ),
        DeclareLaunchArgument(
            'models_dir',
            default_value=default_models_dir,
            description='Path to Gazebo obstacle models directory',
        ),

        Node(
            package='turtlebot3_nav',
            executable='spawn_arena_obstacles',
            name='arena_obstacle_spawner',
            output='screen',
            parameters=[{
                'obstacles_file': obstacles_file,
                'models_dir': models_dir,
                'delete_before_spawn': True,
                'reference_frame': 'world',
            }],
        ),
    ])
