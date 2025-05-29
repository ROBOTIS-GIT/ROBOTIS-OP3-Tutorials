#!/usr/bin/env python3
"""
OP3 Advanced Ball Detector

Real-time ball detection system based on YOLO + OpenVINO
- Automatic device optimization (Intel GPU/CPU)
- Configuration file-based model management
- Real-time performance monitoring

Author: ROBOTIS
Version: 2.0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
from op3_ball_detector_msgs.msg import CircleSetStamped

import cv2
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OpenVINO or YOLO module not found: {e}")
    OPENVINO_AVAILABLE = False


class OP3AdvancedDetector(Node):
    """Advanced object detector for ROBOTIS OP3"""
    
    SUPPORTED_MODELS = {'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'}
    DEFAULT_INPUT_SIZE = (320, 320)
    
    # === Class constants ===
    BALL_CLASS_ID = 32  # COCO sports ball class
    DEFAULT_CONF_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.5
    MAX_DETECTIONS = 3  # Maximum number of balls to detect for better performance
    
    def __init__(self):
        super().__init__('op3_advanced_detector')
        
        # === Initialization sequence ===
        self._init_parameters()
        self._setup_environment()
        self._init_ros_interfaces()
        self._init_detection_model()
        self._init_performance_tracking()
        
        self.get_logger().info(f"✅ Ball Detector ready | {self.device_info}")

    # ==================== Initialization methods ====================
    
    def _init_parameters(self) -> None:
        """Initialize ROS2 parameters and configuration"""
        # Declare ROS2 parameters
        params = [
            ('yolo_model', 'yolov8s'),
            ('openvino_precision', 'FP32'),
            ('camera_topic', '/usb_cam_node/image_raw'),
            ('confidence_threshold', 0.25),
            ('iou_threshold', 0.5),
            ('input_size', [320, 320]),
            ('frame_skip', 2),
            ('debug_mode', True),
            ('enable_performance_log', True)
        ]
        
        for name, default in params:
            self.declare_parameter(name, default)
        
        # Load parameter values
        self.yolo_model_name = self.get_parameter('yolo_model').value
        self.precision = self.get_parameter('openvino_precision').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        input_size_param = self.get_parameter('input_size').value
        self.frame_skip = self.get_parameter('frame_skip').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.enable_performance_log = self.get_parameter('enable_performance_log').value
        
        # Model validation
        if self.yolo_model_name not in self.SUPPORTED_MODELS:
            self.get_logger().warn(f"❌ Unsupported model: {self.yolo_model_name} → using yolov8s")
            self.yolo_model_name = 'yolov8s'
        
        # Input size validation and conversion
        if isinstance(input_size_param, list) and len(input_size_param) == 2:
            self.input_size = tuple(input_size_param)
        else:
            self.get_logger().warn(f"❌ Invalid input_size format: {input_size_param} → using default")
            self.input_size = self.DEFAULT_INPUT_SIZE
        
        # Threshold validation
        self.confidence_threshold = max(0.1, min(0.9, self.confidence_threshold))
        self.iou_threshold = max(0.1, min(0.9, self.iou_threshold))
        self.frame_skip = max(1, self.frame_skip)
        
        self.get_logger().info(f"📋 Configuration complete | Model: {self.yolo_model_name} | Precision: {self.precision}")
        self.get_logger().info(f"📋 Detection settings | Conf: {self.confidence_threshold} | IoU: {self.iou_threshold} | Input: {self.input_size}")
        self.get_logger().info(f"📋 Performance settings | Frame skip: {self.frame_skip} | Performance log: {self.enable_performance_log}")

    def _setup_environment(self) -> None:
        """System environment optimization"""
        # Multithreading optimization
        env_vars = {
            'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 
            'NUMEXPR_NUM_THREADS': '2',
            'OV_CACHE_DIR': '/tmp/ov_cache'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value

    def _init_ros_interfaces(self) -> None:
        """Initialize ROS2 communication interfaces"""
        self.bridge = CvBridge()
        
        # Detect image topic type and create appropriate subscriber
        self._setup_image_subscriber()
        
        # Publishers - compatible with original op3_ball_detector
        self.circle_pub = self.create_publisher(CircleSetStamped, '/ball_detector_node/circle_set', 10)
        self.status_pub = self.create_publisher(String, '/ball_detector_node/status', 10)
        
        # Debug publishers
        self.debug_pub = (
            self.create_publisher(Image, '/ball_detector_node/image_out', 10) 
            if self.debug_mode else None
        )
        
        # Additional publishers for backward compatibility
        self.ball_pub = self.create_publisher(Point, '/ball_position', 10)

    def _setup_image_subscriber(self) -> None:
        """Setup image subscriber based on topic type (compressed or raw)"""
        # Check if the topic is compressed
        if 'compressed' in self.camera_topic.lower():
            self.get_logger().info(f"📹 Using compressed image topic: {self.camera_topic}")
            self.image_sub = self.create_subscription(
                CompressedImage, self.camera_topic, self.compressed_image_callback, 10
            )
        else:
            self.get_logger().info(f"📹 Using raw image topic: {self.camera_topic}")
            self.image_sub = self.create_subscription(
                Image, self.camera_topic, self.image_callback, 10
            )

    def _init_detection_model(self) -> None:
        """Initialize ball detection model"""
        self.device_info = "Unknown"
        self.is_openvino = False
        
        # Try OpenVINO
        if OPENVINO_AVAILABLE and self._try_openvino_setup():
            return
            
        # PyTorch fallback
        self._setup_pytorch_model()

    def _try_openvino_setup(self) -> bool:
        """Attempt OpenVINO model setup"""
        try:
            # Set model file paths
            model_dir = Path(f"{self.yolo_model_name}_openvino_model")
            xml_path = model_dir / f"{self.yolo_model_name}.xml"
            
            # Create model if needed
            if not xml_path.exists():
                self.get_logger().info(f"⚙️  Creating OpenVINO model... ({self.yolo_model_name})")
                self._create_openvino_model()
                
                # Re-check after creation
                if not xml_path.exists():
                    self.get_logger().warn(f"❌ OpenVINO model creation failed: {xml_path}")
                    return False
            
            # Initialize OpenVINO core
            self.ov_core = ov.Core()
            
            # Select device and compile model
            device = self._select_best_device()
            self.ov_model = self.ov_core.read_model(str(xml_path))
            self.compiled_model = self.ov_core.compile_model(self.ov_model, device)
            
            self.device_info = f"OpenVINO {device}"
            self.is_openvino = True
            
            # Warmup
            self._warmup_model()
            
            self.get_logger().info(f"🚀 OpenVINO model loaded | Device: {device}")
            return True
            
        except Exception as e:
            self.get_logger().warn(f"⚠️  OpenVINO setup failed: {e}")
            return False

    def _select_best_device(self) -> str:
        """Select optimal inference device"""
        try:
            available_devices = self.ov_core.available_devices
            
            # Prefer Intel GPU
            if 'GPU' in available_devices:
                return 'GPU'
            elif 'CPU' in available_devices:
                return 'CPU'
            else:
                return 'AUTO'
        except:
            return 'CPU'  # Safe fallback

    def _create_openvino_model(self) -> None:
        """Create OpenVINO optimized model"""
        try:
            # Load PyTorch model
            yolo_model = YOLO(f"{self.yolo_model_name}.pt")
            
            # Export to OpenVINO format
            export_path = yolo_model.export(
                format="openvino",
                half=(self.precision == "FP16"),
                imgsz=self.input_size[0],
                dynamic=False,
                simplify=True
            )
            
            self.get_logger().info(f"✅ OpenVINO model created: {export_path}")
            
        except Exception as e:
            self.get_logger().error(f"❌ OpenVINO model creation failed: {e}")
            raise

    def _warmup_model(self) -> None:
        """Model warmup (remove initial latency)"""
        dummy_input = np.zeros((1, 3, *self.input_size), dtype=np.float32)
        
        try:
            for _ in range(2):
                infer_request = self.compiled_model.create_infer_request()
                infer_request.infer([dummy_input])
            self.get_logger().info("🔥 Model warmup complete")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed: {e}")

    def _setup_pytorch_model(self) -> None:
        """Setup PyTorch YOLO model (fallback)"""
        try:
            self.yolo_model = YOLO(f"{self.yolo_model_name}.pt")
            self.yolo_model.to('cpu')
            self.device_info = "PyTorch CPU"
            self.get_logger().warn("⚠️  PyTorch fallback mode (performance limited)")
        except Exception as e:
            self.get_logger().error(f"❌ Model loading failed: {e}")
            raise

    def _init_performance_tracking(self) -> None:
        """Initialize performance monitoring system"""
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
        self.process_times = []
        self.ball_lost_count = 0

    # ==================== Image processing methods ====================
    
    def compressed_image_callback(self, msg: CompressedImage) -> None:
        """Compressed image processing callback"""
        self.frame_count += 1
        
        # Frame skipping (performance optimization)
        if self.frame_count % self.frame_skip != 0:
            return
            
        start_time = time.time()
        
        try:
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("❌ Failed to decode compressed image")
                return
            
            # Process image using existing pipeline
            processed_img, transform_info = self._preprocess_image(cv_image)

            # Store original frame for debug visualization
            self._current_frame = processed_img.copy()

            # Ball detection
            detections = self._detect_balls(processed_img)
            
            # Publish results
            self._publish_results(detections, transform_info, cv_image.shape)
            
            # Performance tracking
            self._update_performance(time.time() - start_time, len(detections))
            
        except Exception as e:
            self.get_logger().error(f"❌ Compressed image processing failed: {e}")

    def image_callback(self, msg: Image) -> None:
        """Main image processing loop"""
        self.frame_count += 1
        
        # Frame skipping (performance optimization)
        if self.frame_count % self.frame_skip != 0:
            return
            
        start_time = time.time()
        
        try:
            # Image preprocessing
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            processed_img, transform_info = self._preprocess_image(cv_image)

            # Store original frame for debug visualization
            self._current_frame = processed_img.copy()

            # Ball detection
            detections = self._detect_balls(processed_img)
            
            # Publish results
            self._publish_results(detections, transform_info, cv_image.shape)
            
            # Performance tracking
            self._update_performance(time.time() - start_time, len(detections))
            
        except Exception as e:
            self.get_logger().error(f"❌ Image processing failed: {e}")

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Image preprocessing and return transformation info"""
        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Resize maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Center padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, target_h - new_h - pad_h, 
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        transform_info = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'original_size': (w, h)
        }
        
        return padded, transform_info

    def _detect_balls(self, image: np.ndarray) -> List[Dict]:
        """Execute ball detection"""
        try:
            if self.is_openvino:
                return self._openvino_inference(image)
            else:
                return self._pytorch_inference(image)
        except Exception as e:
            self.get_logger().error(f"❌ Detection failed: {e}")
            return []

    def _openvino_inference(self, image: np.ndarray) -> List[Dict]:
        """OpenVINO inference"""
        # Input format conversion: HWC → CHW, BGR → RGB, normalize
        input_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)
        
        # Execute inference
        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer([input_tensor])
        output = infer_request.get_output_tensor(0).data
        
        # Post-processing
        return self._postprocess_detections(output, image.shape)

    def _pytorch_inference(self, image: np.ndarray) -> List[Dict]:
        """PyTorch YOLO inference (fallback)"""
        results = self.yolo_model.predict(
            image,
            classes=[self.BALL_CLASS_ID],
            conf=self.DEFAULT_CONF_THRESHOLD,
            iou=self.DEFAULT_IOU_THRESHOLD,
            verbose=False,
            save=False,
            device='cpu'
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    })
        
        # Sort by confidence
        return sorted(detections, key=lambda x: x['confidence'], reverse=True)

    def _postprocess_detections(self, output: np.ndarray, img_shape: Tuple) -> List[Dict]:
        """OpenVINO output post-processing"""
        detections = []
        
        # YOLOv8 output format: (1, 84, 8400)
        predictions = output[0].T  # (8400, 84)
        
        # Confidence filtering
        ball_scores = predictions[:, 4 + self.BALL_CLASS_ID]
        valid_mask = ball_scores > self.DEFAULT_CONF_THRESHOLD
        
        if not np.any(valid_mask):
            return detections
        
        valid_preds = predictions[valid_mask]
        valid_scores = ball_scores[valid_mask]
        
        # Bounding box conversion (center coordinates → corner coordinates)
        boxes = valid_preds[:, :4]
        x_centers, y_centers = boxes[:, 0], boxes[:, 1]
        widths, heights = boxes[:, 2], boxes[:, 3]
        
        x1s = x_centers - widths / 2
        y1s = y_centers - heights / 2
        x2s = x_centers + widths / 2
        y2s = y_centers + heights / 2
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), valid_scores.tolist(),
            self.DEFAULT_CONF_THRESHOLD, self.DEFAULT_IOU_THRESHOLD
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
                conf = valid_scores[i]
                
                # Minimum size filter
                if (x2 - x1) > 8 and (y2 - y1) > 8:
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    })
        
        # Limit to MAX_DETECTIONS
        return sorted(detections, key=lambda x: x['confidence'], reverse=True)[:self.MAX_DETECTIONS]

    # ==================== Result publishing methods ====================
    
    def _publish_results(self, detections: List[Dict], transform_info: Dict, original_shape: Tuple) -> None:
        """Publish detection results"""
        try:
            # Publish CircleSetStamped message (main output)
            self._publish_circle_set(detections, transform_info, original_shape)
            
            if detections:
                # Publish first ball position for backward compatibility
                self._publish_ball_position(detections[0], transform_info, original_shape)
                self._publish_status(f"DETECTED:conf={detections[0]['confidence']:.3f}")
                self.ball_lost_count = 0
            else:
                self._handle_no_detection()
            
            # Debug image (optional)
            if self.debug_pub and self.debug_pub.get_subscription_count() > 0:
                self._publish_debug_image(detections, transform_info)
                
        except Exception as e:
            self.get_logger().error(f"❌ Result publishing failed: {e}")

    def _publish_circle_set(self, detections: List[Dict], transform_info: Dict, original_shape: Tuple) -> None:
        """Publish CircleSetStamped message compatible with original op3_ball_detector"""
        circle_msg = CircleSetStamped()
        circle_msg.header.stamp = self.get_clock().now().to_msg()
        circle_msg.header.frame_id = "camera_frame"
        
        circles = []
        for detection in detections:
            center_x, center_y = detection['center']
            bbox = detection['bbox']
            
            # Coordinate inverse transformation: remove padding → restore scale
            orig_x = int((center_x - transform_info['pad_w']) / transform_info['scale'])
            orig_y = int((center_y - transform_info['pad_h']) / transform_info['scale'])
            
            # Calculate radius from bounding box
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            radius = int(max(bbox_width, bbox_height) / 2 / transform_info['scale'])
            
            # Boundary clipping
            orig_w, orig_h = transform_info['original_size']
            orig_x = np.clip(orig_x, 0, orig_w - 1)
            orig_y = np.clip(orig_y, 0, orig_h - 1)
            
            # NORMALIZE COORDINATES TO (-1, +1) RANGE
            # top(-1), bottom(+1) for y-axis
            # left(-1), right(+1) for x-axis
            normalized_x = orig_x / orig_w * 2 - 1  # Convert to (-1, +1)
            normalized_y = orig_y / orig_h * 2 - 1  # Convert to (-1, +1)
            
            # Create Point message (x, y are normalized coordinates, z is radius)
            point = Point()
            point.x = float(normalized_x)  # Normalized x (-1 ~ +1)
            point.y = float(normalized_y)  # Normalized y (-1 ~ +1)
            point.z = float(radius)        # radius in pixels
            circles.append(point)
        
        circle_msg.circles = circles
        self.circle_pub.publish(circle_msg)

    def _publish_ball_position(self, detection: Dict, transform_info: Dict, original_shape: Tuple) -> None:
        """Publish ball position information"""
        center_x, center_y = detection['center']
        
        # Coordinate inverse transformation: remove padding → restore scale
        orig_x = int((center_x - transform_info['pad_w']) / transform_info['scale'])
        orig_y = int((center_y - transform_info['pad_h']) / transform_info['scale'])
        
        # Boundary clipping
        orig_w, orig_h = transform_info['original_size']
        orig_x = np.clip(orig_x, 0, orig_w - 1)
        orig_y = np.clip(orig_y, 0, orig_h - 1)
        
        # NORMALIZE COORDINATES TO (-1, +1) RANGE
        normalized_x = orig_x / orig_w * 2 - 1  # Convert to (-1, +1)
        normalized_y = orig_y / orig_h * 2 - 1  # Convert to (-1, +1)
        
        # Publish
        point = Point()
        point.x = float(normalized_x)  # Normalized x (-1 ~ +1)
        point.y = float(normalized_y)  # Normalized y (-1 ~ +1)
        point.z = detection['confidence']
        self.ball_pub.publish(point)

    def _publish_status(self, status: str) -> None:
        """Publish status message"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _handle_no_detection(self) -> None:
        """Handle no ball detection"""
        self.ball_lost_count += 1
        if self.ball_lost_count > 10:  # No detection for 10 consecutive frames
            self._publish_status("NO_BALL")

    def _publish_debug_image(self, detections: List[Dict], transform_info: Dict) -> None:
        """Publish debug image with detection visualizations"""
        try:
            # Use original frame as base for debug image
            if hasattr(self, '_current_frame') and self._current_frame is not None:
                # Resize debug image to match detection input size for consistency
                debug_img = cv2.resize(self._current_frame, self.input_size)
            else:
                # Fallback: create empty image with detection input size
                debug_img = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)
            
            img_height, img_width = debug_img.shape[:2]
            
            # Calculate appropriate text scale based on image size (smaller for 320x320)
            text_scale = 0.4  # Fixed smaller scale for 320x320 resolution
            text_thickness = 1
            
            # Draw detection results directly on resized debug image (no coordinate transformation needed)
            for i, detection in enumerate(detections):
                # Use detection coordinates directly since debug image is same size as detection input
                center_x, center_y = detection['center']
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                center_x = max(0, min(center_x, img_width - 1))
                center_y = max(0, min(center_y, img_height - 1))
                
                # Color coding by detection order
                color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for best, yellow for others
                
                # Draw bounding box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point
                cv2.circle(debug_img, (center_x, center_y), 4, color, -1)
                
                # Draw confidence text
                conf_text = f"Ball {i+1}: {detection['confidence']:.3f}"
                text_y = y1 - 10 if y1 > 20 else y2 + 20
                cv2.putText(debug_img, conf_text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness)
            
            # Add performance info
            fps_text = f"FPS: {self.current_fps:.1f} | Device: {self.device_info}"
            cv2.putText(debug_img, fps_text, (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
            
            # Convert and publish
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = "camera_frame"
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Debug image publishing failed: {e}")

    # ==================== Performance monitoring ====================
    
    def _update_performance(self, process_time: float, ball_count: int) -> None:
        """Update performance statistics"""
        self.process_times.append(process_time)
        self.fps_counter += 1
        
        # Periodic logging (every 15 frames)
        if self.fps_counter % 15 == 0:
            self._log_performance(ball_count)

    def _log_performance(self, ball_count: int) -> None:
        """Log performance information"""
        elapsed = time.time() - self.fps_timer
        self.current_fps = 15 / elapsed
        
        recent_times = self.process_times[-15:]
        avg_ms = np.mean(recent_times) * 1000
        max_ms = np.max(recent_times) * 1000
        
        self.get_logger().info(
            f"📊 {self.device_info} | FPS: {self.current_fps:.1f} | "
            f"Process time: {avg_ms:.0f}ms (max: {max_ms:.0f}ms) | Balls: {ball_count}"
        )
        
        # Reset timer
        self.fps_timer = time.time()
        
        # Memory management
        if len(self.process_times) > 30:
            self.process_times = self.process_times[-15:]


def main(args=None):
    """Main execution function"""
    rclpy.init(args=args)
    
    try:
        detector = OP3AdvancedDetector()
        detector.get_logger().info("🎯 Ball Detector running... (Ctrl+C to stop)")
        rclpy.spin(detector)
        
    except KeyboardInterrupt:
        print("\n👋 User terminated")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
