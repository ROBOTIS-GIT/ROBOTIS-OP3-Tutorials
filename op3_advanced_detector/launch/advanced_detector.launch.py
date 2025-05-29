#!/usr/bin/env python3
"""
YOLO-based ball detection system launch configuration

- Configuration file-based parameter management

Author: ROBOTIS
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch configuration"""
    
    # Package path
    pkg_share = FindPackageShare('op3_advanced_detector')
    
    # Default configuration file path
    default_config_path = PathJoinSubstitution([
        pkg_share, 'config', 'detector_config.yaml'
    ])
    
    return LaunchDescription([
        # Configuration file path argument
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_path,
            description='Ball detector configuration file path (YAML)'
        ),
        
        # Ball detector node
        Node(
            package='op3_advanced_detector',
            executable='advanced_detector',
            name='advanced_detector',
            parameters=[LaunchConfiguration('config_file')],
            output='screen',
            emulate_tty=True,  # Color log output support
        )
    ])