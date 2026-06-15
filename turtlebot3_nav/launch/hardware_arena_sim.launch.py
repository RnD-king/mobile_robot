import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_turtlebot3_nav = get_package_share_directory('turtlebot3_nav')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    default_world = os.path.join(
        pkg_turtlebot3_nav,
        'worlds',
        'hardware_arena.world',
    )
    default_obstacles_file = os.path.join(
        pkg_turtlebot3_nav,
        'config',
        'arena_obstacles.yaml',
    )
    default_models_dir = os.path.join(
        pkg_turtlebot3_nav,
        'models',
    )

    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    map_profile = LaunchConfiguration('map')
    spawn_obstacles = LaunchConfiguration('spawn_obstacles')
    start_cartographer = LaunchConfiguration('start_cartographer')
    start_nav = LaunchConfiguration('start_nav')
    obstacles_file = LaunchConfiguration('obstacles_file')
    models_dir = LaunchConfiguration('models_dir')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)
    goal_x_param = ParameterValue(goal_x, value_type=float)
    goal_y_param = ParameterValue(goal_y, value_type=float)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world,
            'verbose': 'false',
        }.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_turtlebot3_gazebo,
                'launch',
                'robot_state_publisher.launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    spawn_turtlebot3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_turtlebot3_gazebo,
                'launch',
                'spawn_turtlebot3.launch.py'
            )
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )

    obstacle_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='turtlebot3_nav',
                executable='spawn_arena_obstacles',
                name='arena_obstacle_spawner',
                output='screen',
                condition=IfCondition(spawn_obstacles),
                parameters=[{
                    'obstacles_file': obstacles_file,
                    'models_dir': models_dir,
                    'delete_before_spawn': True,
                    'reference_frame': 'world',
                }],
            ),
        ],
    )

    cartographer = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        pkg_turtlebot3_nav,
                        'launch',
                        'cartographer_no_return_free.launch.py',
                    )
                ),
                condition=IfCondition(start_cartographer),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'use_rviz': use_rviz,
                    'map': map_profile,
                }.items(),
            ),
        ],
    )

    hardware_nav = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='turtlebot3_nav',
                executable='turtlebot_nav_hardware',
                name='turtlebot_nav_hardware',
                output='screen',
                condition=IfCondition(start_nav),
                parameters=[{
                    'use_sim_time': use_sim_time_param,
                    'goal_x': goal_x_param,
                    'goal_y': goal_y_param,
                }],
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Path to custom Gazebo world file',
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='Initial TurtleBot3 x position',
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='Initial TurtleBot3 y position',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Start RViz through the Cartographer launch',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='world',
            description='Cartographer profile for this arena: world or house',
        ),
        DeclareLaunchArgument(
            'spawn_obstacles',
            default_value='true',
            description='Spawn arena obstacles from YAML',
        ),
        DeclareLaunchArgument(
            'start_cartographer',
            default_value='true',
            description='Start Cartographer SLAM and occupancy grid publisher',
        ),
        DeclareLaunchArgument(
            'start_nav',
            default_value='true',
            description='Start turtlebot_nav_hardware node',
        ),
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
        DeclareLaunchArgument(
            'goal_x',
            default_value='5.5',
            description='Goal x in the initial robot frame',
        ),
        DeclareLaunchArgument(
            'goal_y',
            default_value='0.0',
            description='Goal y in the initial robot frame',
        ),

        gazebo,
        robot_state_publisher,
        spawn_turtlebot3,
        obstacle_spawner,
        cartographer,
        hardware_nav,
    ])
