# OP3 Advanced Ball Detector

🎯 **High-performance real-time ball detection system based on YOLO + OpenVINO**

## ✨ Key Features

- **🚀 Auto Optimization**: Automatic Intel GPU/CPU detection and optimization
- **⚙️ Easy Configuration**: All settings managed through YAML file
- **📊 Real-time Monitoring**: FPS and processing performance tracking
- **🎯 Accurate Detection**: High accuracy with YOLO v8 + OpenVINO combination
- **🔄 Smart Recovery**: Automatic error recovery and model reloading
- **⚡ Performance Optimized**: Maximum 3 ball detections for better speed
- **🛡️ Robust Error Handling**: Graceful handling of missing dependencies and errors

## 🏆 Performance Comparison

| Model | Speed | Accuracy | Memory | Recommended Use |
|-------|-------|----------|--------|-----------------|
| **yolov8n** | ⚡⚡⚡ | ⭐⭐ | 💾 | Low-spec hardware |
| **yolov8s** | ⚡⚡ | ⭐⭐⭐ | 💾💾 | **Default recommended** |
| **yolov8m** | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 | Accuracy priority |
| **yolov8l** | 🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | High-performance system |
| **yolov8x** | 🐌🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾💾 | Research/benchmarking |

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
# Method 1: Using conda (recommended)
conda activate yolo_env
pip install ultralytics openvino

# Method 2: Using system Python
pip install ultralytics openvino

# Optional: Intel GPU support
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero
```

### 2. Download YOLO Model (Optional)
```bash
# Models will be auto-downloaded, but you can pre-download:
cd ~/robotis_ws/src
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

### 3. Build Package and Run
```bash
cd ~/robotis_ws
colcon build --packages-select op3_advanced_detector
source install/setup.bash

# Run with default configuration
ros2 launch op3_advanced_detector advanced_detector.launch.py

# Run with custom configuration
ros2 launch op3_advanced_detector advanced_detector.launch.py config_file:=/path/to/custom_config.yaml
```

## ⚙️ Configuration

### Main Configuration File: `config/detector_config.yaml`

```yaml
/**:
  ros__parameters:
    # === Model Configuration ===
    yolo_model: "yolov8s"                    # Model: n, s, m, l, x
    model_path: ""                           # Custom model path (optional)
    input_size: [320, 320]                   # Input resolution [width, height]
    
    # === OpenVINO Configuration ===
    openvino_precision: "FP32"               # Precision: FP32, FP16
    openvino_device: "AUTO"                  # Device: AUTO, GPU, CPU
    use_openvino: true                       # Enable OpenVINO optimization
    
    # === Detection Parameters ===
    confidence_threshold: 0.25               # Confidence threshold (0.0-1.0)
    iou_threshold: 0.5                       # IoU threshold for NMS
    max_detections: 3                        # Maximum balls to detect
    
    # === Camera Configuration ===
    camera_topic: "/usb_cam_node/image_raw"  # Input camera topic
    
    # === Debug and Monitoring ===
    debug_mode: true                         # Enable debug image output
    performance_monitoring: true             # Enable FPS monitoring
    log_level: "INFO"                        # Log level: DEBUG, INFO, WARN, ERROR
    
    # === Advanced Settings ===
    processing_interval: 0.1                # Processing interval (seconds)
    error_recovery_enabled: true             # Enable automatic error recovery
    max_error_count: 5                       # Max consecutive errors before restart
```

### Alternative Configuration Methods

1. **Environment Variables**:
```bash
export YOLO_MODEL=yolov8m
export OPENVINO_DEVICE=GPU
ros2 launch op3_advanced_detector advanced_detector.launch.py
```

2. **Launch Parameters**:
```bash
ros2 launch op3_advanced_detector advanced_detector.launch.py \
    yolo_model:=yolov8m \
    openvino_device:=GPU \
    debug_mode:=false
```

## 📡 ROS2 Interface

### Published Topics
- `/ball_position` (geometry_msgs/Point): Ball position (x, y, confidence)
- `/detection_status` (std_msgs/String): Detection status and performance info
- `/detection_debug` (sensor_msgs/Image): Debug image with bounding boxes (optional)

### Subscribed Topics
- `/usb_cam_node/image_raw` (sensor_msgs/Image): Input camera image

### Services
- `/reset_detector` (std_srvs/Empty): Reset detector and reload model

## 📊 Performance Monitoring

### Real-time Performance Logs
```
[INFO] 📋 Configuration complete | Model: yolov8s | Precision: FP32 | Max detections: 3
[INFO] 🚀 OpenVINO model loaded | Device: GPU | Optimization: FP32
[INFO] ✅ Ball Detector ready | Backend: OpenVINO GPU
[INFO] 🎯 Ball Detector running... (Ctrl+C to exit)
[INFO] 📊 OpenVINO GPU | FPS: 18.5 | Processing: 38ms (Max: 65ms) | Balls: 2/3
```

### Performance Metrics
- **FPS**: Frames processed per second
- **Processing Time**: Time per frame (current/maximum)
- **Ball Count**: Number of detected balls (current/maximum)
- **Device Usage**: Current processing device (GPU/CPU)

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04 LTS or later
- **ROS**: ROS2 Foxy or later
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS2 Jazzy
- **Python**: 3.10+
- **RAM**: 8GB
- **GPU**: Intel integrated graphics or dedicated GPU
- **Storage**: 4GB free space

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Import Errors (ultralytics/openvino not found)**
```bash
# Check current environment
which python
pip list | grep ultralytics

# Install missing packages
pip install ultralytics openvino
```

**2. OpenVINO GPU Not Available**
```bash
# Install Intel GPU drivers
sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero

# Verify GPU detection
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

**3. Poor Performance**
- Use smaller model: `yolov8n` instead of `yolov8m`
- Enable FP16 precision: `openvino_precision: "FP16"`
- Reduce input size: `input_size: [320, 320]`
- Lower confidence threshold: `confidence_threshold: 0.15`

**4. Camera Connection Issues**
```bash
# Check camera topics
ros2 topic list | grep image

# Test camera
ros2 topic echo /usb_cam_node/image_raw --field header
```

**5. High Memory Usage**
- Reduce `max_detections` to 1 or 2
- Use `yolov8n` model
- Enable `openvino_precision: "FP16"`

### Error Recovery Features

The detector includes automatic error recovery:
- **Model reload**: Automatic model reloading on errors
- **Device fallback**: GPU → CPU fallback on device errors
- **Connection retry**: Automatic camera reconnection
- **Memory cleanup**: Automatic memory management

## 🚀 Performance Optimization Guide

### 1. Hardware Optimization
```yaml
# For Intel GPU systems
openvino_device: "GPU"
openvino_precision: "FP16"
use_openvino: true

# For CPU-only systems
openvino_device: "CPU"
openvino_precision: "FP32"
max_detections: 1
```

### 2. Model Selection Guide
- **Real-time applications**: `yolov8n` or `yolov8s`
- **Balanced performance**: `yolov8s` (default)
- **High accuracy needs**: `yolov8m` or `yolov8l`
- **Research/benchmarking**: `yolov8x`

### 3. Memory Optimization
```yaml
input_size: [320, 320]     # Smaller input size
max_detections: 3          # Limit detection count
openvino_precision: "FP16" # Half precision
```

### 4. Processing Optimization
```yaml
confidence_threshold: 0.3   # Higher threshold = fewer detections
processing_interval: 0.1    # Process every 100ms
debug_mode: false          # Disable debug output
```

## 🧪 Testing and Validation

### Unit Testing
```bash
cd ~/robotis_ws
colcon test --packages-select op3_advanced_detector
```

### Performance Benchmarking
```bash
# Test different models
for model in yolov8n yolov8s yolov8m; do
    ros2 launch op3_advanced_detector advanced_detector.launch.py \
        yolo_model:=$model debug_mode:=false
done
```

### Debug Mode Testing
```bash
# Enable detailed logging and debug output
ros2 launch op3_advanced_detector advanced_detector.launch.py \
    debug_mode:=true log_level:=DEBUG
```

## 📄 License

Apache 2.0 License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on the repository
- Check the troubleshooting section above
- Review the performance optimization guide

---

**💡 Developer Notes**: 
- This system is optimized for Intel hardware but works on any x86_64 system
- Maximum 3 ball detections provide the best balance of accuracy and performance
- OpenVINO GPU acceleration can provide 2-3x performance improvement over CPU-only execution
- The system includes comprehensive error handling and automatic recovery mechanisms