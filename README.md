ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py 
gzclient
ros2 launch turtlebot3_nav cartographer_no_return_free.launch.py
ros2 launch turtlebot3_nav turtlebot_nav_sim.launch.py

ros2 topic pub --once /move_base_simple/goal geometry_msgs/msg/PoseStamped "
header:
  frame_id: map
pose:
  position:
    x: 6.0
    y: 0.0
    z: 0.0
  orientation:
    w: 1.0
"
