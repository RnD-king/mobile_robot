import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'turtlebot3_nav'


def package_files(directory):
    paths = []
    if not os.path.exists(directory):
        return paths

    for path, _, files in os.walk(directory):
        file_paths = [
            os.path.join(path, filename)
            for filename in files
        ]
        if file_paths:
            install_path = os.path.join('share', package_name, path)
            paths.append((install_path, file_paths))
    return paths


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            'share/' + package_name + '/launch',
            glob('launch/*.launch.py')
        ),
        (
            'share/' + package_name + '/worlds',
            glob('worlds/*.world')
        ),
        (
            'share/' + package_name + '/config',
            glob('config/*.yaml')
        ),
        (
            'share/' + package_name + '/rviz',
            glob('rviz/*.rviz')
        ),
    ] + package_files('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noh',
    maintainer_email='noh@example.com',
    description='Custom online map navigation node for TurtleBot3 Burger without Nav2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'turtlebot_nav_sim = turtlebot3_nav.turtlebot_nav_sim:main',
            'turtlebot_nav_hardware = turtlebot3_nav.turtlebot_nav_hardware:main',
            'scan_no_return_filter = turtlebot3_nav.scan_no_return_filter:main',
            'spawn_arena_obstacles = turtlebot3_nav.spawn_arena_obstacles:main',
        ],
    },
)
