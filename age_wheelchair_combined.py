import ultralytics.utils.patches as patches
import pathlib
import os
import sys
import torch
import cv2
import numpy as np
from deepface import DeepFace
from pathlib import Path
import glob
from mtcnn import MTCNN  # 새로 추가

# ultralytics 내부 _torch_load 함수 패치
_original_torch_load = patches.torch_load
def patched_torch_load(f, *args, **kwargs):
    if isinstance(f, (pathlib.PosixPath, pathlib.WindowsPath)):
        f = str(f)
    return _original_torch_load(f, *args, **kwargs)
patches._torch_load = patched_torch_load

# YOLOv5 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

weights_path = str(FILE.parents[0] / "best_fixed.pt")
device = torch.device("cpu")
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt

detector = MTCNN()  # MTCNN 초기화

def analyze_age(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
        return int(result[0]['age'])
    except Exception as e:
        print(f"DeepFace analyze_age error: {e}")
        return None

def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

def process_frame(frame):
    img0 = frame.copy()
    img = letterbox(img0, 640, stride=stride, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    wheelchair_detected = False
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred:
            label = names[int(cls)]
            if label in ['wheelchair', 'peopleWithWheelchair', 'person_wheelchair']:
                wheelchair_detected = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)
    print(f"Detected faces by MTCNN: {len(results)}")

    age_detected = None
    for face in results:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_roi = frame_rgb[y:y+h, x:x+w]
        try:
            age = analyze_age(face_roi)
            if age:
                age_detected = age
                print(f"Detected age: {age_detected}")
                break
        except Exception as e:
            print(f"DeepFace analyze_age error: {e}")

    wheelchair_text = "Wheelchair detected: YES" if wheelchair_detected else "Wheelchair detected: NO"
    is_vulnerable = wheelchair_detected or (age_detected is not None and age_detected >= 65)
    status_text = "Vulnerable: YES" if is_vulnerable else "Vulnerable: NO"

    print(wheelchair_text)
    print(status_text)

    return True

source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

if source == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break
        if not process_frame(frame):
            break
    cap.release()

elif os.path.isdir(source):
    image_files = sorted([f for f in glob.glob(os.path.join(source, '*')) if is_image_file(f)])
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            continue
        if not process_frame(frame):
            break

elif os.path.isfile(source):
    if is_image_file(source):
        frame = cv2.imread(source)
        if frame is None:
            print(f"이미지를 불러올 수 없습니다: {source}")
            sys.exit(1)
        process_frame(frame)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"동영상 파일을 열 수 없습니다: {source}")
            sys.exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not process_frame(frame):
                break
        cap.release()

else:
    print(f"입력 소스를 인식하지 못했습니다: {source}")
    sys.exit(1)
