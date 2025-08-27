# README.txt

프로젝트명: 휠체어 및 노인 인식 통합 시스템

설명:
- YOLOv5 기반의 휠체어 객체 탐지와 MTCNN + DeepFace를 활용한 얼굴 나이 인식을 결합한 통합 인식 프로그램입니다.
- 두 조건 중 하나라도 만족하면 'Vulnerable: YES'로 판단합니다.
- CPU 환경에서 동작하며, 기본적으로 웹캠, 이미지, 동영상 파일을 입력받아 처리할 수 있습니다.
- 본 프로젝트는 Windows Subsystem for Linux (WSL) Ubuntu 20.04 환경에서 개발 및 테스트 되었습니다.

필요 환경:
- Python 3.8 이상 권장
- WSL Ubuntu 20.04 이상 권장
- 필수 패키지: ultralytics, torch, torchvision, opencv-python, numpy, deepface, mtcnn 등
- 패키지 설치: `pip install -r requirements`

파일 설명:
- age_wheelchair_combined.py : 메인 실행 스크립트
- best_fixed.pt             : YOLOv5 휠체어 탐지 모델 가중치
- yolov5/                   : YOLOv5 모델 소스 코드 및 관련 라이브러리
- haarcascade_frontalface_default.xml : (Haar Cascade 사용 시 필요) 얼굴 검출용 파일
- requirements             : 필요한 Python 패키지 목록

실행 방법 예시:
1) 웹캠 실시간 인식  
   python3 age_wheelchair_combined.py webcam

2) 이미지 파일 인식  
   python3 age_wheelchair_combined.py path/to/image.jpg

3) 이미지 폴더 인식  
   python3 age_wheelchair_combined.py path/to/image_folder

4) 동영상 파일 인식  
   python3 age_wheelchair_combined.py path/to/video.mp4

출력문 설명:
- `Detected faces by MTCNN: N`  
  -> 현재 프레임(또는 이미지)에서 MTCNN이 인식한 얼굴 개수

- `Detected age: XX`  
  -> 검출된 대표 얼굴의 나이 (DeepFace 분석 결과)

- `Wheelchair detected: YES/NO`  
  -> 현재 프레임에서 YOLOv5가 휠체어 객체를 탐지했는지 여부

- `Vulnerable: YES/NO`  
  -> 휠체어 탐지 또는 나이가 65세 이상일 경우 YES, 아니면 NO 출력

참고:
- MTCNN을 이용해 얼굴 인식 성능을 개선했으며, 나이 인식은 DeepFace를 사용합니다.
- 모델 정확도 개선 및 속도 향상을 위해 별도 학습 및 GPU 활용이 가능합니다.
- WSL 환경에서 실행 시 경로 설정과 권한 문제에 유의하세요.

