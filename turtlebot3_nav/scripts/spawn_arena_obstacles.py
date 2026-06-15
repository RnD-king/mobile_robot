#!/usr/bin/env python3

import math
import os
from typing import Dict, Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.node import Node


class ArenaObstacleSpawner(Node):
    def __init__(self):
        super().__init__('arena_obstacle_spawner')

        package_share = get_package_share_directory('turtlebot3_nav')
        default_obstacles_file = os.path.join(
            package_share, 'config', 'arena_obstacles.yaml'
        )
        default_models_dir = os.path.join(package_share, 'models')

        self.declare_parameter('obstacles_file', default_obstacles_file)
        self.declare_parameter('models_dir', default_models_dir)
        self.declare_parameter('delete_before_spawn', True)
        self.declare_parameter('reference_frame', 'world')

        self.obstacles_file = self.get_parameter('obstacles_file').value
        self.models_dir = self.get_parameter('models_dir').value
        self.delete_before_spawn = bool(
            self.get_parameter('delete_before_spawn').value
        )
        self.reference_frame = self.get_parameter('reference_frame').value

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

    def wait_for_services(self) -> bool:
        self.get_logger().info('Waiting for Gazebo spawn/delete services...')
        if not self.spawn_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service /spawn_entity is not available.')
            return False
        if self.delete_before_spawn:
            if not self.delete_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().error('Service /delete_entity is not available.')
                return False
        return True

    def load_config(self) -> Dict:
        if not os.path.exists(self.obstacles_file):
            raise FileNotFoundError(f'Cannot find {self.obstacles_file}')

        with open(self.obstacles_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        if 'obstacles' not in data or not isinstance(data['obstacles'], list):
            raise ValueError('YAML must contain an obstacles: list')

        return data

    def model_xml(self, obstacle_type: str) -> str:
        sdf_path = os.path.join(self.models_dir, obstacle_type, 'model.sdf')
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f'Cannot find model SDF: {sdf_path}')

        with open(sdf_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def make_pose(x: float, y: float, z: float, yaw_rad: float) -> Pose:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.z = math.sin(yaw_rad * 0.5)
        pose.orientation.w = math.cos(yaw_rad * 0.5)
        return pose

    def delete_entity(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        # Deleting a non-existing entity can fail; this is fine because we spawn next.

    def spawn_entity(
        self,
        name: str,
        xml: str,
        pose: Pose,
        robot_namespace: str = '',
    ) -> bool:
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.robot_namespace = robot_namespace
        req.initial_pose = pose
        req.reference_frame = self.reference_frame

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is None:
            self.get_logger().error(f'Failed to call /spawn_entity for {name}')
            return False

        result = future.result()
        if not result.success:
            self.get_logger().error(f'Failed to spawn {name}: {result.status_message}')
            return False

        self.get_logger().info(f'Spawned {name}: {result.status_message}')
        return True

    def spawn_all(self) -> int:
        data = self.load_config()
        count = 0

        for obs in data['obstacles']:
            if not obs.get('enabled', True):
                continue

            name = str(obs['name'])
            obstacle_type = str(obs['type'])
            x = float(obs.get('x', 0.0))
            y = float(obs.get('y', 0.0))
            z = float(obs.get('z', 0.0))

            if 'yaw_rad' in obs:
                yaw_rad = float(obs['yaw_rad'])
            else:
                yaw_rad = math.radians(float(obs.get('yaw_deg', 0.0)))

            xml = self.model_xml(obstacle_type)
            pose = self.make_pose(x, y, z, yaw_rad)

            if self.delete_before_spawn:
                self.delete_entity(name)

            if self.spawn_entity(name, xml, pose):
                count += 1

        return count


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = ArenaObstacleSpawner()
    try:
        if node.wait_for_services():
            count = node.spawn_all()
            node.get_logger().info(f'Done. Spawned {count} obstacle(s).')
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
