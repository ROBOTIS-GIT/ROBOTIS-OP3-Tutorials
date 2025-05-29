# OP3 Advanced Ball Detector 튜토리얼 📚

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [환경 설정](#2-환경-설정)
3. [패키지 생성 과정](#3-패키지-생성-과정)
4. [코드 구조 및 설명](#4-코드-구조-및-설명)
5. [빌드 및 실행](#5-빌드-및-실행)
6. [사용법 및 설정](#6-사용법-및-설정)
7. [문제 해결](#7-문제-해결)

---

## 1. 프로젝트 소개

### 1.1 프로젝트 개요
`op3_advanced_detector`는 ROBOTIS OP3 로봇을 위한 **현대적인 AI 기반 공 감지 시스템**입니다. 기존의 OpenCV 기반 감지 방식을 **YOLO (You Only Look Once) 딥러닝 모델**로 대체하여 더 정확하고 빠른 감지 성능을 제공합니다.

### 1.2 주요 특징
- 🚀 **YOLO v8 딥러닝 모델** 사용으로 높은 정확도
- ⚡ **OpenVINO 최적화**로 실시간 처리 성능
- 🔧 **ROS2 호환**으로 기존 OP3 시스템과 완벽 통합
- 📊 **실시간 성능 모니터링** 및 디버그 기능
- 🎛️ **YAML 설정 파일**로 쉬운 파라미터 관리

### 1.3 시스템 요구사항
- **운영체제**: Ubuntu 22.04 LTS
- **ROS**: ROS2 Humble
- **Python**: 3.10+
- **하드웨어**: ROBOTIS OP3 + USB 카메라

---

## 2. 환경 설정

### 2.1 필수 의존성 설치

#### Python 패키지 설치
```bash
# YOLO 모델 라이브러리
pip install ultralytics

# OpenVINO 최적화 (선택사항, 성능 향상)
pip install openvino

# 이미지 처리 라이브러리
pip install opencv-python numpy
```

#### ROS2 의존성 설치
```bash
# 작업 공간으로 이동
cd ~/robotis_ws

# ROS 의존성 자동 설치
rosdep install --from-paths src --ignore-src -r -y
```

### 2.2 환경 변수 설정
```bash
# .bashrc에 추가
echo "source ~/robotis_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 3. 패키지 생성 과정

### 3.1 ROS2 Python 패키지 생성
```bash
# 작업 공간의 src 디렉토리로 이동
cd ~/robotis_ws/src/tutorials

# ROS2 Python 패키지 생성
ros2 pkg create --build-type ament_python op3_advanced_detector \
  --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge op3_ball_detector_msgs
```

### 3.2 패키지 구조 설정
```bash
cd op3_advanced_detector

# 필요한 디렉토리 생성
mkdir -p config launch resource

# Python 모듈 디렉토리 생성
mkdir -p op3_advanced_detector
```

### 3.3 기본 파일 생성

#### package.xml 설정
```xml
<?xml version="1.0"?>
<package format="3">
  <name>op3_advanced_detector</name>
  <version>0.0.1</version>
  <description>
    OP3 Advanced Ball Detector - YOLO + OpenVINO based real-time ball detection system.
  </description>
  <maintainer email="robotis@robotis.com">ROBOTIS</maintainer>
  <license>Apache 2.0</license>

  <!-- 핵심 의존성 -->
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>cv_bridge</depend>
  <depend>op3_ball_detector_msgs</depend>
  
  <!-- 실행 의존성 -->
  <exec_depend>usb_cam</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

#### setup.py 설정
```python
from setuptools import setup
import os
from glob import glob

package_name = 'op3_advanced_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ROBOTIS',
    maintainer_email='robotis@robotis.com',
    description='Advanced ball detector for ROBOTIS-OP3',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'advanced_detector = op3_advanced_detector.op3_advanced_detector:main',
        ],
    },
)
```

---

## 4. 코드 구조 및 설명

### 4.1 메인 클래스 구조

#### OP3AdvancedDetector 클래스
```python
class OP3AdvancedDetector(Node):
    """OP3용 고급 객체 감지기"""
    
    # 클래스 상수
    SUPPORTED_MODELS = {'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'}
    BALL_CLASS_ID = 32  # COCO 데이터셋의 스포츠 공 클래스
    DEFAULT_CONF_THRESHOLD = 0.25  # 신뢰도 임계값
```

### 4.2 핵심 기능 설명

#### 📥 이미지 전처리 (`_preprocess_image`)
```python
def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    이미지를 YOLO 입력 형식으로 변환
    
    1. 종횡비 유지하며 크기 조정
    2. 패딩을 추가해 정사각형으로 만들기
    3. 변환 정보 저장 (역변환용)
    """
```

#### 🧠 AI 모델 추론 (`_detect_balls`)
```python
def _detect_balls(self, image: np.ndarray) -> List[Dict]:
    """
    YOLO 모델을 사용한 공 감지
    
    - OpenVINO 가속: GPU/CPU 자동 선택
    - PyTorch 백업: 호환성 보장
    """
```

#### 📡 결과 발행 (`_publish_results`)
```python
def _publish_results(self, detections: List[Dict], transform_info: Dict, original_shape: Tuple):
    """
    감지 결과를 ROS 메시지로 발행
    
    1. CircleSetStamped: 기존 OP3 시스템 호환
    2. Point: 공 위치 정보
    3. 디버그 이미지: 시각화
    """
```

### 4.3 좌표 변환 시스템

#### 정규화된 좌표계
```python
# 화면 좌표를 (-1, +1) 범위로 정규화
normalized_x = orig_x / orig_w * 2 - 1  # 좌(-1) ~ 우(+1)
normalized_y = orig_y / orig_h * 2 - 1  # 위(-1) ~ 아래(+1)
```

**왜 정규화를 할까요?**
- 화면 해상도에 무관하게 일관된 좌표 제공
- 로봇의 모션 제어에서 표준화된 입력 사용 가능

### 4.4 성능 최적화 기법

#### OpenVINO 가속
```python
def _try_openvino_setup(self) -> bool:
    """
    OpenVINO로 모델 최적화
    
    1. 모델을 Intel 최적화 형식으로 변환
    2. GPU/CPU 자동 선택
    3. 추론 속도 2-3배 향상
    """
```

#### 프레임 스키핑
```python
# 성능 최적화: N번째 프레임만 처리
if self.frame_count % self.frame_skip != 0:
    return
```

---

## 5. 빌드 및 실행

### 5.1 패키지 빌드

#### 전체 워크스페이스 빌드
```bash
cd ~/robotis_ws

# 의존성 설치
rosdep install --from-paths src --ignore-src -r -y

# 빌드 실행
colcon build --packages-select op3_advanced_detector

# 환경 설정 적용
source install/setup.bash
```

#### 빌드 확인
```bash
# 패키지 설치 확인
ros2 pkg list | grep op3_advanced_detector

# 실행 파일 확인
ros2 run op3_advanced_detector advanced_detector --help
```

### 5.2 실행 방법

#### 기본 실행
```bash
# USB 카메라 노드 실행 (터미널 1)
ros2 run usb_cam usb_cam_node_exe

# 볼 감지기 실행 (터미널 2)
ros2 run op3_advanced_detector advanced_detector
```

#### Launch 파일 실행
```bash
# 모든 노드 동시 실행
ros2 launch op3_advanced_detector ball_detector_from_usb_cam.launch.py

# 설정 파일 지정 실행
ros2 launch op3_advanced_detector advanced_detector.launch.py \
  config_file:=~/my_config.yaml
```

### 5.3 실행 확인

#### 토픽 모니터링
```bash
# 감지된 공 정보 확인
ros2 topic echo /ball_detector_node/circle_set

# 공 위치 정보 확인
ros2 topic echo /ball_position

# 시스템 상태 확인
ros2 topic echo /ball_detector_node/status
```

#### 디버그 이미지 확인
```bash
# RViz2에서 시각화
rviz2

# 또는 rqt에서 이미지 확인
rqt_image_view /ball_detector_node/image_out
```

---

## 6. 사용법 및 설정

### 6.1 설정 파일 사용법

#### detector_config.yaml 수정
```yaml
/**:
  ros__parameters:
    # 모델 선택 (성능 vs 정확도)
    yolo_model: "yolov8s"        # 권장: 균형잡힌 성능
    
    # 감지 민감도 조정
    confidence_threshold: 0.25   # 낮을수록 더 많이 감지
    iou_threshold: 0.5          # 중복 감지 제거 강도
    
    # 성능 최적화
    frame_skip: 2               # 처리 프레임 간격
    input_size: [320, 320]      # 입력 이미지 크기
    
    # 디버그 설정
    debug_mode: true            # 시각화 활성화
```

#### 모델별 특성 비교

| 모델 | 속도 | 정확도 | 메모리 사용량 | 권장 용도 |
|------|------|--------|---------------|-----------|
| yolov8n | ⭐⭐⭐⭐⭐ | ⭐⭐ | 낮음 | 저사양 시스템 |
| yolov8s | ⭐⭐⭐⭐ | ⭐⭐⭐ | 보통 | **일반 권장** |
| yolov8m | ⭐⭐⭐ | ⭐⭐⭐⭐ | 높음 | 고정확도 필요 |
| yolov8l | ⭐⭐ | ⭐⭐⭐⭐⭐ | 매우 높음 | 연구/개발용 |

### 6.2 카메라 설정

#### USB 카메라 설정
```yaml
# camera_param.yaml
usb_cam_node:
  ros__parameters:
    video_device: "/dev/video0"
    image_width: 640
    image_height: 480
    pixel_format: "yuyv"
    framerate: 30.0
```

#### 카메라 캘리브레이션
```bash
# 카메라 정보 확인
v4l2-ctl --list-devices

# 해상도 및 프레임레이트 확인
v4l2-ctl --list-formats-ext
```

### 6.3 성능 튜닝 가이드

#### 실시간 성능 모니터링
```bash
# 로그에서 성능 정보 확인
ros2 run op3_advanced_detector advanced_detector 2>&1 | grep "📊"

# 예시 출력:
# 📊 OpenVINO GPU | FPS: 28.5 | Process time: 35ms (max: 48ms) | Balls: 1
```

#### 성능 최적화 팁

1. **저사양 시스템**
   ```yaml
   yolo_model: "yolov8n"
   input_size: [224, 224]
   frame_skip: 3
   ```

2. **고성능 시스템**
   ```yaml
   yolo_model: "yolov8m"
   input_size: [480, 480]
   frame_skip: 1
   ```

3. **안정성 우선**
   ```yaml
   confidence_threshold: 0.4
   openvino_precision: "FP32"
   ```

---

## 7. 문제 해결

### 7.1 일반적인 문제

#### Q1: "ModuleNotFoundError: No module named 'ultralytics'"
**해결책:**
```bash
pip install ultralytics
# 또는 conda 환경의 경우
conda install -c conda-forge ultralytics
```

#### Q2: OpenVINO 설치 실패
**해결책:**
```bash
# OpenVINO 수동 설치
pip install openvino-dev[pytorch]

# 또는 PyTorch 모드로 실행 (성능 제한)
# detector_config.yaml에서 openvino_precision을 주석 처리
```

#### Q3: 카메라가 인식되지 않음
**해결책:**
```bash
# 카메라 장치 확인
ls /dev/video*

# 권한 설정
sudo usermod -a -G video $USER
sudo reboot

# USB 포트 변경 후 재시도
```

#### Q4: 감지 성능이 떨어짐
**해결책:**
1. **조명 환경 개선**: 충분한 조명 확보
2. **임계값 조정**: `confidence_threshold` 값 낮추기
3. **모델 업그레이드**: 더 큰 모델 사용
4. **해상도 증가**: `input_size` 크기 늘리기

### 7.2 디버깅 방법

#### 로그 레벨 조정
```bash
# 상세 로그 출력
ros2 run op3_advanced_detector advanced_detector --ros-args --log-level DEBUG
```

#### 시각적 디버깅
```bash
# 디버그 이미지 확인
rqt_image_view /ball_detector_node/image_out

# 토픽 그래프 확인
rqt_graph
```

#### 성능 프로파일링
```python
# 성능 측정 코드 추가
import time
start_time = time.time()
# ... 코드 실행 ...
print(f"처리 시간: {(time.time() - start_time)*1000:.2f}ms")
```

### 7.3 고급 문제 해결

#### OpenVINO 모델 변환 문제
```bash
# 수동 모델 변환
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='openvino', imgsz=320)
"
```

#### 메모리 부족 문제
```yaml
# 설정 최적화
input_size: [224, 224]  # 더 작은 입력 크기
frame_skip: 3           # 더 많은 프레임 스키핑
yolo_model: "yolov8n"   # 더 작은 모델
```

---

## 마무리

이 튜토리얼을 통해 **OP3 Advanced Ball Detector** 패키지를 성공적으로 구축하고 실행할 수 있었습니다. 이 시스템은 기존 OpenCV 기반 감지기 대비 다음과 같은 개선점을 제공합니다:

### ✅ 주요 개선사항
- **정확도 향상**: YOLO v8 딥러닝 모델 사용
- **성능 최적화**: OpenVINO 가속 및 프레임 스키핑
- **설정 편의성**: YAML 기반 파라미터 관리
- **호환성**: 기존 OP3 시스템과 완벽 통합
- **확장성**: 다양한 객체 감지로 확장 가능

### 🚀 다음 단계
- **다중 객체 감지**: 공 외에 다른 객체 추가 감지
- **트래킹 기능**: 객체 추적 알고리즘 통합
- **실시간 학습**: 환경 적응형 모델 개발
- **모바일 최적화**: 더 작고 빠른 모델 적용

### 📞 지원 및 문의
- **GitHub Issues**: 버그 리포트 및 기능 제안
- **ROBOTIS Forum**: 커뮤니티 지원
- **Documentation**: 공식 문서 및 API 레퍼런스

Happy Coding! 🤖⚽
