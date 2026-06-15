from glob import glob

from setuptools import find_packages, setup

package_name = 'turtlebot3_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noh',
    maintainer_email='noh@example.com',
    description='Custom online map navigation node for TurtleBot3 Burger without Nav2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'turtlebot_nav_sim = turtlebot3_nav.turtlebot_nav_sim:main',
            'scan_no_return_filter = turtlebot3_nav.scan_no_return_filter:main',
        ],
    },
)
