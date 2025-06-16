#!/bin/bash

# Activate virtual environment
source ~/yolo_env/bin/activate

# Source ROS2 environment
source ~/robotis_ws/install/setup.bash

# Start voice control system
echo "Starting OP3 Voice Control System..."
ros2 launch op3_voice_control voice_control.launch.py

# Cleanup on exit
echo "Voice control system stopped."