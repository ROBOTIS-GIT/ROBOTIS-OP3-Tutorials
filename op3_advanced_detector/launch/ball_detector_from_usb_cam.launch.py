"""
Launch file for OP3 Advanced Ball Detector with USB Camera
This replaces the original op3_ball_detector launch file with YOLO-based detection
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    ld = LaunchDescription()
    
    # Use op3_advanced_detector package path
    pkg_share = FindPackageShare('op3_advanced_detector')
    camera_param_path = PathJoinSubstitution([pkg_share, 'config', 'camera_param.yaml'])
    
    # USB Camera Node (same as original)
    usb_cam_node = Node(
        package='usb_cam', 
        namespace='usb_cam_node',
        executable='usb_cam_node_exe',
        output='screen',
        parameters=[camera_param_path],
    )

    # Launch argument for config file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'detector_config.yaml']),
        description='Ball detector configuration file path (YAML)'
    )

    # Advanced Ball Detector Node
    ball_detector_node = Node(
        package='op3_advanced_detector',
        executable='advanced_detector',
        name='advanced_detector',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        # Remappings for image input
        remappings=[
            ('/ball_detector_node/image_in', '/usb_cam_node/image_raw/compressed'),
            ('/ball_detector_node/cameraInfo_in', '/usb_cam_node/camera_info'),
        ],
        emulate_tty=True,  # Color log output support
        prefix='/home/robotis/yolo_env/bin/python3'  # Use Python virtual environment for YOLO
    )
    
    # Add all actions
    ld.add_action(config_file_arg)
    ld.add_action(usb_cam_node)
    ld.add_action(ball_detector_node)

    return ld
