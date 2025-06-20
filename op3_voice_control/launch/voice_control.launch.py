#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Parameter declarations
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.expanduser('~/op3_voice_tutorial/models/vosk-model-small-en-us'),
        description='Path to Vosk model'
    )
    
    device_id_arg = DeclareLaunchArgument(
        'device_id',
        default_value='c920_mic',
        description='Audio device ID (c920_mic for default)'
    )
    
    # Speech recognition node
    speech_recognition_node = Node(
        package='op3_voice_control',
        executable='speech_recognition_node',
        name='speech_recognition_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'device_id': LaunchConfiguration('device_id'),
            'sample_rate': 16000,
            'buffer_size': 4096,
            'channels': 1,
            'noise_reduction': True
        }],
        output='screen',
        prefix='/home/robotis/yolo_env/bin/python3'
    )
    
    # Command processor node
    command_processor_node = Node(
        package='op3_voice_control',
        executable='command_processor',
        name='command_processor',
        output='screen',
        prefix='/home/robotis/yolo_env/bin/python3'
    )
    
    return LaunchDescription([
        model_path_arg,
        device_id_arg,
        speech_recognition_node,
        command_processor_node
    ])