# OP3 Advanced Detector 함수별 상세 튜토리얼 🔍

## 목차
1. [클래스 구조 개요](#1-클래스-구조-개요)
2. [초기화 함수들](#2-초기화-함수들)
3. [이미지 처리 함수들](#3-이미지-처리-함수들)
4. [AI 모델 추론 함수들](#4-ai-모델-추론-함수들)
5. [결과 발행 함수들](#5-결과-발행-함수들)
6. [성능 모니터링 함수들](#6-성능-모니터링-함수들)
7. [유틸리티 함수들](#7-유틸리티-함수들)

---

## 1. 클래스 구조 개요

### 1.1 OP3AdvancedDetector 클래스
```python
class OP3AdvancedDetector(Node):
    """ROBOTIS OP3용 고급 객체 감지기"""
```

**주요 특징:**
- ROS2 Node를 상속받아 ROS 시스템과 통합
- YOLO v8 + OpenVINO 기반 실시간 공 감지
- 성능 최적화 및 모니터링 기능 내장

**클래스 상수:**
```python
SUPPORTED_MODELS = {'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'}
BALL_CLASS_ID = 32  # COCO 데이터셋의 스포츠 공 클래스
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 3
```

---

## 2. 초기화 함수들

### 2.1 `__init__(self)` - 메인 생성자
```python
def __init__(self):
    super().__init__('op3_advanced_detector')
    
    # 초기화 시퀀스
    self._init_parameters()
    self._setup_environment()
    self._init_ros_interfaces()
    self._init_detection_model()
    self._init_performance_tracking()
```

**역할:**
- ROS2 노드 초기화
- 5단계 초기화 프로세스 실행
- 시스템 준비 완료 메시지 출력

**초기화 순서가 중요한 이유:**
1. **파라미터 먼저**: 다른 모든 설정의 기준
2. **환경 설정**: 성능 최적화를 위한 시스템 변수
3. **ROS 인터페이스**: 통신 채널 구축
4. **AI 모델**: 가장 시간이 오래 걸리는 작업
5. **성능 추적**: 실행 후 모니터링 준비

### 2.2 `_init_parameters(self)` - ROS2 파라미터 초기화
```python
def _init_parameters(self) -> None:
    """ROS2 파라미터 및 설정 초기화"""
    
    # 파라미터 선언
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
```

**주요 기능:**
- **파라미터 선언**: ROS2 파라미터 시스템에 등록
- **값 검증**: 입력값의 유효성 검사
- **기본값 설정**: 안전한 기본 설정 제공

**파라미터별 상세 설명:**

| 파라미터 | 기본값 | 설명 | 영향 |
|----------|--------|------|------|
| `yolo_model` | 'yolov8s' | 사용할 YOLO 모델 | 정확도 vs 속도 |
| `confidence_threshold` | 0.25 | 감지 신뢰도 임계값 | 민감도 조절 |
| `input_size` | [320, 320] | 입력 이미지 크기 | 성능 vs 정확도 |
| `frame_skip` | 2 | 처리 프레임 간격 | CPU 사용률 |

### 2.3 `_setup_environment(self)` - 시스템 환경 최적화
```python
def _setup_environment(self) -> None:
    """시스템 환경 최적화"""
    env_vars = {
        'OMP_NUM_THREADS': '2',      # OpenMP 스레드 수 제한
        'MKL_NUM_THREADS': '2',      # Intel MKL 스레드 수 제한
        'NUMEXPR_NUM_THREADS': '2',  # NumExpr 스레드 수 제한
        'OV_CACHE_DIR': '/tmp/ov_cache'  # OpenVINO 캐시 디렉토리
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
```

**최적화 목적:**
- **메모리 사용량 제한**: 다중 스레드로 인한 메모리 폭증 방지
- **성능 안정화**: 스레드 경합 최소화
- **캐시 관리**: OpenVINO 모델 캐시 위치 지정

### 2.4 `_init_ros_interfaces(self)` - ROS2 통신 인터페이스 설정
```python
def _init_ros_interfaces(self) -> None:
    """ROS2 통신 인터페이스 초기화"""
    
    self.bridge = CvBridge()  # OpenCV ↔ ROS 이미지 변환
    
    # 이미지 구독자 설정 (압축/비압축 자동 감지)
    self._setup_image_subscriber()
    
    # 발행자들 설정
    self.circle_pub = self.create_publisher(CircleSetStamped, '/ball_detector_node/circle_set', 10)
    self.status_pub = self.create_publisher(String, '/ball_detector_node/status', 10)
    self.ball_pub = self.create_publisher(Point, '/ball_position', 10)
    
    # 디버그 발행자 (조건부)
    self.debug_pub = (
        self.create_publisher(Image, '/ball_detector_node/image_out', 10) 
        if self.debug_mode else None
    )
```

**토픽 구조:**
- **입력**: `/usb_cam_node/image_raw` (카메라 이미지)
- **출력**: 
  - `/ball_detector_node/circle_set` (감지된 공 정보)
  - `/ball_position` (첫 번째 공 위치)
  - `/ball_detector_node/status` (시스템 상태)
  - `/ball_detector_node/image_out` (디버그 이미지)

### 2.5 `_init_detection_model(self)` - AI 모델 초기화
```python
def _init_detection_model(self) -> None:
    """공 감지 모델 초기화"""
    self.device_info = "Unknown"
    self.is_openvino = False
    
    # OpenVINO 우선 시도
    if OPENVINO_AVAILABLE and self._try_openvino_setup():
        return
        
    # PyTorch 백업
    self._setup_pytorch_model()
```

**모델 선택 우선순위:**
1. **OpenVINO (Intel 최적화)**: 최고 성능
2. **PyTorch CPU**: 호환성 백업

---

## 3. 이미지 처리 함수들

### 3.1 `image_callback(self, msg)` - 메인 이미지 처리 루프
```python
def image_callback(self, msg: Image) -> None:
    """메인 이미지 처리 루프"""
    self.frame_count += 1
    
    # 프레임 스키핑 (성능 최적화)
    if self.frame_count % self.frame_skip != 0:
        return
        
    start_time = time.time()
    
    try:
        # 1. 이미지 전처리
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        processed_img, transform_info = self._preprocess_image(cv_image)

        # 2. 디버그용 프레임 저장
        self._current_frame = processed_img.copy()

        # 3. 공 감지 실행
        detections = self._detect_balls(processed_img)
        
        # 4. 결과 발행
        self._publish_results(detections, transform_info, cv_image.shape)
        
        # 5. 성능 추적
        self._update_performance(time.time() - start_time, len(detections))
        
    except Exception as e:
        self.get_logger().error(f"❌ Image processing failed: {e}")
```

**처리 파이프라인:**
```mermaid
graph LR
    A[ROS Image] --> B[OpenCV 변환]
    B --> C[전처리]
    C --> D[AI 추론]
    D --> E[후처리]
    E --> F[ROS 발행]
```

**프레임 스키핑의 효과:**
- `frame_skip=2`: 30fps → 15fps (CPU 사용량 50% 감소)
- `frame_skip=3`: 30fps → 10fps (CPU 사용량 67% 감소)

### 3.2 `_preprocess_image(self, image)` - 이미지 전처리
```python
def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """이미지 전처리 및 변환 정보 반환"""
    h, w = image.shape[:2]
    target_w, target_h = self.input_size
    
    # 1. 종횡비 유지하며 크기 조정
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # 2. 중앙 패딩으로 정사각형 만들기
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_h - new_h - pad_h, 
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)  # 회색 패딩
    )
    
    # 3. 변환 정보 저장 (역변환용)
    transform_info = {
        'scale': scale,
        'pad_w': pad_w,
        'pad_h': pad_h,
        'original_size': (w, h)
    }
    
    return padded, transform_info
```

**전처리가 필요한 이유:**
- **YOLO 요구사항**: 정사각형 입력 필요
- **종횡비 유지**: 이미지 왜곡 방지
- **표준화**: 다양한 해상도 지원

**변환 과정 시각화:**
```
원본 (640x480) → 리사이즈 (320x240) → 패딩 (320x320)
[    이미지    ]   [  이미지  ]      [  이미지  ]
                                     [   패딩   ]
```

### 3.3 `compressed_image_callback(self, msg)` - 압축 이미지 처리
```python
def compressed_image_callback(self, msg: CompressedImage) -> None:
    """압축 이미지 처리 콜백"""
    # ... 프레임 스키핑 로직 동일 ...
    
    try:
        # JPEG 압축 해제
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            self.get_logger().error("❌ Failed to decode compressed image")
            return
        
        # 기존 파이프라인 재사용
        processed_img, transform_info = self._preprocess_image(cv_image)
        # ... 나머지 처리 동일 ...
```

**압축 이미지 지원 이유:**
- **대역폭 절약**: 네트워크 전송 최적화
- **호환성**: 다양한 카메라 타입 지원

---

## 4. AI 모델 추론 함수들

### 4.1 `_detect_balls(self, image)` - 공 감지 실행
```python
def _detect_balls(self, image: np.ndarray) -> List[Dict]:
    """공 감지 실행"""
    try:
        if self.is_openvino:
            return self._openvino_inference(image)  # 고성능 경로
        else:
            return self._pytorch_inference(image)   # 백업 경로
    except Exception as e:
        self.get_logger().error(f"❌ Detection failed: {e}")
        return []
```

**이중 백업 시스템:**
- **1차**: OpenVINO (Intel 최적화)
- **2차**: PyTorch (범용 호환)

### 4.2 `_openvino_inference(self, image)` - OpenVINO 추론
```python
def _openvino_inference(self, image: np.ndarray) -> List[Dict]:
    """OpenVINO 추론"""
    
    # 1. 입력 형식 변환: HWC → CHW, BGR → RGB, 정규화
    input_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, 0)  # 배치 차원 추가
    
    # 2. 추론 실행
    infer_request = self.compiled_model.create_infer_request()
    infer_request.infer([input_tensor])
    output = infer_request.get_output_tensor(0).data
    
    # 3. 후처리
    return self._postprocess_detections(output, image.shape)
```

**OpenVINO 최적화 효과:**
- **속도**: PyTorch 대비 2-3배 빠름
- **메모리**: 효율적인 메모리 사용
- **하드웨어**: Intel GPU/CPU 전용 최적화

**입력 변환 과정:**
```python
# 원본: (320, 320, 3) BGR uint8
# ↓ 색상 변환
# (320, 320, 3) RGB uint8
# ↓ 축 변환 및 정규화
# (3, 320, 320) RGB float32 [0-1]
# ↓ 배치 차원 추가
# (1, 3, 320, 320) RGB float32 [0-1]
```

### 4.3 `_pytorch_inference(self, image)` - PyTorch 추론 (백업)
```python
def _pytorch_inference(self, image: np.ndarray) -> List[Dict]:
    """PyTorch YOLO 추론 (백업)"""
    results = self.yolo_model.predict(
        image,
        classes=[self.BALL_CLASS_ID],  # 공(32번) 클래스만 감지
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
    
    return sorted(detections, key=lambda x: x['confidence'], reverse=True)
```

**PyTorch 모드 특징:**
- **간편함**: 고수준 API 사용
- **안정성**: 더 많은 에러 처리
- **호환성**: 다양한 하드웨어 지원

### 4.4 `_postprocess_detections(self, output, img_shape)` - 감지 결과 후처리
```python
def _postprocess_detections(self, output: np.ndarray, img_shape: Tuple) -> List[Dict]:
    """OpenVINO 출력 후처리"""
    detections = []
    
    # 1. YOLOv8 출력 형식: (1, 84, 8400)
    predictions = output[0].T  # (8400, 84)로 전치
    
    # 2. 신뢰도 필터링
    ball_scores = predictions[:, 4 + self.BALL_CLASS_ID]  # 공 클래스 점수
    valid_mask = ball_scores > self.DEFAULT_CONF_THRESHOLD
    
    if not np.any(valid_mask):
        return detections
    
    valid_preds = predictions[valid_mask]
    valid_scores = ball_scores[valid_mask]
    
    # 3. 바운딩 박스 변환 (중심 좌표 → 모서리 좌표)
    boxes = valid_preds[:, :4]
    x_centers, y_centers = boxes[:, 0], boxes[:, 1]
    widths, heights = boxes[:, 2], boxes[:, 3]
    
    x1s = x_centers - widths / 2
    y1s = y_centers - heights / 2
    x2s = x_centers + widths / 2
    y2s = y_centers + heights / 2
    
    # 4. NMS (Non-Maximum Suppression) 적용
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), valid_scores.tolist(),
        self.DEFAULT_CONF_THRESHOLD, self.DEFAULT_IOU_THRESHOLD
    )
    
    # 5. 최종 감지 결과 생성
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            conf = valid_scores[i]
            
            # 최소 크기 필터 (노이즈 제거)
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                })
    
    # 6. 상위 N개 결과만 반환 (성능 최적화)
    return sorted(detections, key=lambda x: x['confidence'], reverse=True)[:self.MAX_DETECTIONS]
```

**YOLOv8 출력 구조:**
```
출력 텐서: (1, 84, 8400)
├── 1: 배치 크기
├── 84: [x, y, w, h] + 80개 클래스 점수
└── 8400: 예측 박스 개수 (다양한 크기의 그리드)

클래스 점수 위치:
- 0-3: 바운딩 박스 좌표 (x, y, w, h)
- 4-83: 80개 COCO 클래스 점수
- 36번 인덱스 (4+32): 스포츠 공 클래스
```

**NMS (Non-Maximum Suppression):**
- **목적**: 중복 감지 제거
- **방법**: IoU > 임계값인 박스들 중 낮은 신뢰도 제거
- **효과**: 하나의 공에 대해 하나의 박스만 남김

---

## 5. 결과 발행 함수들

### 5.1 `_publish_results(self, detections, transform_info, original_shape)` - 결과 발행 통합
```python
def _publish_results(self, detections: List[Dict], transform_info: Dict, original_shape: Tuple) -> None:
    """감지 결과 발행"""
    try:
        # 1. 메인 출력: CircleSetStamped 메시지
        self._publish_circle_set(detections, transform_info, original_shape)
        
        if detections:
            # 2. 첫 번째 공 위치 (호환성)
            self._publish_ball_position(detections[0], transform_info, original_shape)
            self._publish_status(f"DETECTED:conf={detections[0]['confidence']:.3f}")
            self.ball_lost_count = 0
        else:
            # 3. 감지 실패 처리
            self._handle_no_detection()
        
        # 4. 디버그 이미지 (선택사항)
        if self.debug_pub and self.debug_pub.get_subscription_count() > 0:
            self._publish_debug_image(detections, transform_info)
            
    except Exception as e:
        self.get_logger().error(f"❌ Result publishing failed: {e}")
```

**발행 우선순위:**
1. **필수**: CircleSetStamped (메인 출력)
2. **호환성**: Point 메시지 (기존 시스템용)
3. **상태**: 시스템 상태 문자열
4. **선택**: 디버그 이미지 (구독자 있을 때만)

### 5.2 `_publish_circle_set(self, detections, transform_info, original_shape)` - 원 정보 발행
```python
def _publish_circle_set(self, detections: List[Dict], transform_info: Dict, original_shape: Tuple) -> None:
    """CircleSetStamped 메시지 발행 (기존 op3_ball_detector 호환)"""
    circle_msg = CircleSetStamped()
    circle_msg.header.stamp = self.get_clock().now().to_msg()
    circle_msg.header.frame_id = "camera_frame"
    
    circles = []
    for detection in detections:
        center_x, center_y = detection['center']
        bbox = detection['bbox']
        
        # 1. 좌표 역변환: 패딩 제거 → 스케일 복원
        orig_x = int((center_x - transform_info['pad_w']) / transform_info['scale'])
        orig_y = int((center_y - transform_info['pad_h']) / transform_info['scale'])
        
        # 2. 바운딩 박스에서 반지름 계산
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        radius = int(max(bbox_width, bbox_height) / 2 / transform_info['scale'])
        
        # 3. 경계 클리핑
        orig_w, orig_h = transform_info['original_size']
        orig_x = np.clip(orig_x, 0, orig_w - 1)
        orig_y = np.clip(orig_y, 0, orig_h - 1)
        
        # 4. 좌표 정규화 (-1 ~ +1 범위)
        normalized_x = orig_x / orig_w * 2 - 1  # 좌(-1) ~ 우(+1)
        normalized_y = orig_y / orig_h * 2 - 1  # 위(-1) ~ 아래(+1)
        
        # 5. Point 메시지 생성
        point = Point()
        point.x = float(normalized_x)  # 정규화된 x 좌표
        point.y = float(normalized_y)  # 정규화된 y 좌표
        point.z = float(radius)        # 픽셀 단위 반지름
        circles.append(point)
    
    circle_msg.circles = circles
    self.circle_pub.publish(circle_msg)
```

**좌표 변환 과정:**
```
1. AI 출력 좌표 (320x320 기준)
   ↓
2. 패딩 제거: x' = x - pad_w, y' = y - pad_h
   ↓
3. 스케일 복원: x'' = x' / scale, y'' = y' / scale
   ↓
4. 원본 해상도 좌표 (640x480 기준)
   ↓
5. 정규화: nx = x'' / width * 2 - 1, ny = y'' / height * 2 - 1
   ↓
6. 정규화된 좌표 (-1 ~ +1 범위)
```

**정규화의 장점:**
- **해상도 독립적**: 어떤 카메라든 동일한 좌표계
- **로봇 제어 표준화**: 머리 움직임 각도와 직접 연결
- **수학적 편의**: 삼각함수 계산 간소화

### 5.3 `_publish_ball_position(self, detection, transform_info, original_shape)` - 공 위치 발행
```python
def _publish_ball_position(self, detection: Dict, transform_info: Dict, original_shape: Tuple) -> None:
    """공 위치 정보 발행"""
    center_x, center_y = detection['center']
    
    # 좌표 역변환 (위와 동일한 과정)
    orig_x = int((center_x - transform_info['pad_w']) / transform_info['scale'])
    orig_y = int((center_y - transform_info['pad_h']) / transform_info['scale'])
    
    # 경계 클리핑
    orig_w, orig_h = transform_info['original_size']
    orig_x = np.clip(orig_x, 0, orig_w - 1)
    orig_y = np.clip(orig_y, 0, orig_h - 1)
    
    # 정규화
    normalized_x = orig_x / orig_w * 2 - 1
    normalized_y = orig_y / orig_h * 2 - 1
    
    # Point 메시지 발행
    point = Point()
    point.x = float(normalized_x)     # 정규화된 x
    point.y = float(normalized_y)     # 정규화된 y
    point.z = detection['confidence'] # 신뢰도
    self.ball_pub.publish(point)
```

**Point 메시지 구조:**
- `x`: 정규화된 x 좌표 (-1 ~ +1)
- `y`: 정규화된 y 좌표 (-1 ~ +1)  
- `z`: 감지 신뢰도 (0 ~ 1)

### 5.4 `_publish_debug_image(self, detections, transform_info)` - 디버그 이미지 발행
```python
def _publish_debug_image(self, detections: List[Dict], transform_info: Dict) -> None:
    """디버그 이미지 발행 (감지 결과 시각화)"""
    try:
        # 1. 베이스 이미지 준비
        if hasattr(self, '_current_frame') and self._current_frame is not None:
            debug_img = cv2.resize(self._current_frame, self.input_size)
        else:
            debug_img = np.zeros((*self.input_size[::-1], 3), dtype=np.uint8)
        
        img_height, img_width = debug_img.shape[:2]
        text_scale = 0.4
        text_thickness = 1
        
        # 2. 감지 결과 그리기
        for i, detection in enumerate(detections):
            center_x, center_y = detection['center']
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # 경계 확인
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            center_x = max(0, min(center_x, img_width - 1))
            center_y = max(0, min(center_y, img_height - 1))
            
            # 색상 지정 (첫 번째는 초록, 나머지는 노랑)
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            
            # 바운딩 박스 그리기
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            
            # 중심점 그리기
            cv2.circle(debug_img, (center_x, center_y), 4, color, -1)
            
            # 신뢰도 텍스트
            conf_text = f"Ball {i+1}: {detection['confidence']:.3f}"
            text_y = y1 - 10 if y1 > 20 else y2 + 20
            cv2.putText(debug_img, conf_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness)
        
        # 3. 성능 정보 추가
        fps_text = f"FPS: {self.current_fps:.1f} | Device: {self.device_info}"
        cv2.putText(debug_img, fps_text, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
        
        # 4. ROS 메시지로 변환 및 발행
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        debug_msg.header.frame_id = "camera_frame"
        self.debug_pub.publish(debug_msg)
        
    except Exception as e:
        self.get_logger().error(f"❌ Debug image publishing failed: {e}")
```

**디버그 이미지 요소:**
- **바운딩 박스**: 감지된 공 영역 (사각형)
- **중심점**: 공의 중심 위치 (원)
- **신뢰도**: 감지 확신도 (텍스트)
- **성능 정보**: FPS 및 사용 장치 (텍스트)
- **색상 코딩**: 첫 번째 공(초록), 나머지(노랑)

---

## 6. 성능 모니터링 함수들

### 6.1 `_update_performance(self, process_time, ball_count)` - 성능 업데이트
```python
def _update_performance(self, process_time: float, ball_count: int) -> None:
    """성능 통계 업데이트"""
    self.process_times.append(process_time)
    self.fps_counter += 1
    
    # 주기적 로깅 (15프레임마다)
    if self.fps_counter % 15 == 0:
        self._log_performance(ball_count)
```

**성능 추적 항목:**
- **처리 시간**: 각 프레임 처리에 걸리는 시간
- **FPS**: 초당 처리 프레임 수
- **감지 개수**: 프레임당 감지된 공 개수

### 6.2 `_log_performance(self, ball_count)` - 성능 로깅
```python
def _log_performance(self, ball_count: int) -> None:
    """성능 정보 로깅"""
    elapsed = time.time() - self.fps_timer
    self.current_fps = 15 / elapsed
    
    recent_times = self.process_times[-15:]
    avg_ms = np.mean(recent_times) * 1000
    max_ms = np.max(recent_times) * 1000
    
    self.get_logger().info(
        f"📊 {self.device_info} | FPS: {self.current_fps:.1f} | "
        f"Process time: {avg_ms:.0f}ms (max: {max_ms:.0f}ms) | Balls: {ball_count}"
    )
    
    # 타이머 리셋
    self.fps_timer = time.time()
    
    # 메모리 관리
    if len(self.process_times) > 30:
        self.process_times = self.process_times[-15:]
```

**로그 예시:**
```
📊 OpenVINO GPU | FPS: 28.5 | Process time: 35ms (max: 48ms) | Balls: 1
📊 PyTorch CPU | FPS: 12.3 | Process time: 81ms (max: 125ms) | Balls: 0
```

**성능 지표 해석:**
- **FPS > 20**: 실시간 처리 가능
- **Process time < 50ms**: 양호한 성능
- **Max time**: 최악의 경우 지연시간

---

## 7. 유틸리티 함수들

### 7.1 OpenVINO 관련 함수들

#### `_try_openvino_setup(self)` - OpenVINO 설정 시도
```python
def _try_openvino_setup(self) -> bool:
    """OpenVINO 모델 설정 시도"""
    try:
        # 1. 모델 파일 경로 설정
        model_dir = Path(f"{self.yolo_model_name}_openvino_model")
        xml_path = model_dir / f"{self.yolo_model_name}.xml"
        
        # 2. 모델 생성 (필요시)
        if not xml_path.exists():
            self.get_logger().info(f"⚙️  Creating OpenVINO model... ({self.yolo_model_name})")
            self._create_openvino_model()
        
        # 3. OpenVINO 코어 초기화
        self.ov_core = ov.Core()
        
        # 4. 최적 장치 선택
        device = self._select_best_device()
        
        # 5. 모델 컴파일
        self.ov_model = self.ov_core.read_model(str(xml_path))
        self.compiled_model = self.ov_core.compile_model(self.ov_model, device)
        
        # 6. 설정 완료
        self.device_info = f"OpenVINO {device}"
        self.is_openvino = True
        
        # 7. 워밍업
        self._warmup_model()
        
        return True
        
    except Exception as e:
        self.get_logger().warn(f"⚠️  OpenVINO setup failed: {e}")
        return False
```

#### `_create_openvino_model(self)` - OpenVINO 모델 생성
```python
def _create_openvino_model(self) -> None:
    """OpenVINO 최적화 모델 생성"""
    try:
        # PyTorch 모델 로드
        yolo_model = YOLO(f"{self.yolo_model_name}.pt")
        
        # OpenVINO 형식으로 내보내기
        export_path = yolo_model.export(
            format="openvino",
            half=(self.precision == "FP16"),  # 반정밀도 사용 여부
            imgsz=self.input_size[0],
            dynamic=False,   # 고정 입력 크기
            simplify=True    # 그래프 최적화
        )
        
        self.get_logger().info(f"✅ OpenVINO model created: {export_path}")
        
    except Exception as e:
        self.get_logger().error(f"❌ OpenVINO model creation failed: {e}")
        raise
```

**모델 변환 옵션:**
- **FP16**: 메모리 사용량 50% 감소, 약간의 정확도 손실
- **FP32**: 높은 정확도, 더 많은 메모리 사용
- **Dynamic=False**: 고정 크기로 더 빠른 추론
- **Simplify=True**: 불필요한 연산 제거

#### `_select_best_device(self)` - 최적 장치 선택
```python
def _select_best_device(self) -> str:
    """최적 추론 장치 선택"""
    try:
        available_devices = self.ov_core.available_devices
        
        # Intel GPU 우선
        if 'GPU' in available_devices:
            return 'GPU'
        elif 'CPU' in available_devices:
            return 'CPU'
        else:
            return 'AUTO'
    except:
        return 'CPU'  # 안전한 백업
```

**장치 우선순위:**
1. **GPU**: Intel 내장 그래픽 (가장 빠름)
2. **CPU**: Intel CPU (중간 성능)
3. **AUTO**: 자동 선택 (안전)

#### `_warmup_model(self)` - 모델 워밍업
```python
def _warmup_model(self) -> None:
    """모델 워밍업 (초기 지연시간 제거)"""
    dummy_input = np.zeros((1, 3, *self.input_size), dtype=np.float32)
    
    try:
        for _ in range(2):
            infer_request = self.compiled_model.create_infer_request()
            infer_request.infer([dummy_input])
        self.get_logger().info("🔥 Model warmup complete")
    except Exception as e:
        self.get_logger().warn(f"Warmup failed: {e}")
```

**워밍업의 필요성:**
- **초기 지연**: 첫 추론은 캐시 로딩 등으로 느림
- **메모리 할당**: GPU 메모리 미리 할당
- **최적화**: 런타임 최적화 완료

### 7.2 에러 처리 및 상태 관리

#### `_handle_no_detection(self)` - 감지 실패 처리
```python
def _handle_no_detection(self) -> None:
    """공 감지 실패 처리"""
    self.ball_lost_count += 1
    if self.ball_lost_count > 10:  # 10프레임 연속 감지 실패
        self._publish_status("NO_BALL")
```

**감지 실패 로직:**
- **카운터 증가**: 연속 실패 횟수 추적
- **임계값 적용**: 10프레임 = 약 0.5초 (30fps 기준)
- **상태 발행**: 다른 노드에 상황 알림

#### `_publish_status(self, status)` - 상태 발행
```python
def _publish_status(self, status: str) -> None:
    """상태 메시지 발행"""
    msg = String()
    msg.data = status
    self.status_pub.publish(msg)
```

**상태 메시지 종류:**
- `"DETECTED:conf=0.856"`: 감지 성공 + 신뢰도
- `"NO_BALL"`: 감지 실패 (장기간)
- `"ERROR:..."`: 시스템 오류

### 7.3 메인 실행 함수

#### `main(args=None)` - 프로그램 진입점
```python
def main(args=None):
    """메인 실행 함수"""
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
```

**실행 흐름:**
1. **ROS 초기화**: `rclpy.init()`
2. **노드 생성**: `OP3AdvancedDetector()`
3. **메인 루프**: `rclpy.spin()` - 콜백 처리
4. **정리**: `rclpy.shutdown()`

---

## 마무리

### 🎯 핵심 함수 요약

| 카테고리 | 주요 함수 | 역할 |
|----------|-----------|------|
| **초기화** | `__init__`, `_init_parameters` | 시스템 설정 및 준비 |
| **이미지 처리** | `image_callback`, `_preprocess_image` | 입력 데이터 처리 |
| **AI 추론** | `_detect_balls`, `_openvino_inference` | 공 감지 실행 |
| **결과 발행** | `_publish_results`, `_publish_circle_set` | 출력 데이터 전송 |
| **성능 모니터링** | `_update_performance`, `_log_performance` | 시스템 상태 추적 |

### 💡 함수 설계 원칙

1. **단일 책임**: 각 함수는 하나의 명확한 목적
2. **에러 안전**: 예외 처리 및 백업 메커니즘
3. **성능 우선**: 최적화된 경로와 백업 경로 제공
4. **확장성**: 새로운 기능 추가 용이
5. **모니터링**: 실시간 성능 추적 및 로깅

### 🚀 활용 방법

이 함수들을 이해하면:
- **커스터마이징**: 특정 요구사항에 맞게 수정
- **디버깅**: 문제 발생 지점 빠른 파악
- **최적화**: 성능 병목 지점 개선
- **확장**: 새로운 객체 감지 기능 추가

Happy Coding! 🤖⚽
