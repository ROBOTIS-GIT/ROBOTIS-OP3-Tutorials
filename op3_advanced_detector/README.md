# OP3 Advanced Ball Detector

AI-powered real-time ball detection system for ROBOTIS OP3 using YOLO v8 and OpenVINO optimization.

## Features

- **High Performance**: YOLO v8 + OpenVINO acceleration
- **Easy Setup**: YAML configuration and automatic model download
- **Real-time Monitoring**: FPS tracking and performance logs
- **Robust**: Automatic error recovery and device fallback

## Quick Start

### 1. Install Dependencies
```bash
pip install ultralytics openvino
```

### 2. Build and Run
```bash
cd ~/robotis_ws
colcon build --packages-select op3_advanced_detector
source install/setup.bash

# Run detector
ros2 launch op3_advanced_detector advanced_detector.launch.py
```

## Configuration

Edit `config/detector_config.yaml`:

```yaml
/**:
  ros__parameters:
    yolo_model: "yolov8s"                    # Model size: n, s, m, l, x
    confidence_threshold: 0.25               # Detection threshold
    input_size: [320, 320]                   # Input resolution
    camera_topic: "/usb_cam_node/image_raw"  # Camera input
    debug_mode: true                         # Enable debug output
```

## ROS2 Topics

- **Input**: `/usb_cam_node/image_raw` (sensor_msgs/Image)
- **Output**: `/ball_detector_node/circle_set` (op3_ball_detector_msgs/CircleSetStamped)
- **Debug**: `/ball_detector_node/image_out` (sensor_msgs/Image)

## Model Performance

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| yolov8n | ⚡⚡⚡ | ⭐⭐ | Low | Low-spec hardware |
| yolov8s | ⚡⚡ | ⭐⭐⭐ | Medium | **Default** |
| yolov8m | ⚡ | ⭐⭐⭐⭐ | High | High accuracy |

## Troubleshooting

**Import errors:**
```bash
pip install ultralytics openvino
```

**Poor performance:**
- Use smaller model (`yolov8n`)
- Enable FP16 precision
- Reduce input size to `[224, 224]`

**Camera issues:**
```bash
ros2 topic list | grep image
```

## License

Apache 2.0