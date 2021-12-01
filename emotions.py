import cv2
import numpy as np
import dlib # conda install -c conda-forge dlib=19.4 사용 요망
from imutils import face_utils
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input

# ERROR 정보
# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
# CUDA 11.0 / tensorflow-gpu 2.4.0 / tensorflow 2.4.0 / keras 2.4.3 사용




USE_WEBCAM = False # True : 웹 캠 사용, False : 비디오 파일 사용





# 데이터와 이미지를 로드 하기 위한 파라미터
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# 얼굴 외곽 테두리 지정
frame_window = 10
emotion_offsets = (20, 40)

# 모델 불러오기
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# 예측을 위한 모델 셰입 지정
emotion_target_size = emotion_classifier.input_shape[1:3]

# 모드 계산 시작 리스트
emotion_window = []

# 웹 캠 비디오 스트리밍 시작
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)


# 웹 캠 또는 비디오 파일
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # 웹 캠
else:
    cap = cv2.VideoCapture('./test/test.mp4') # 비디오 파일

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue



        # 7가지 감정

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0)) # 분노, 빨강
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255)) # 슬픔, 파랑
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0)) # 기쁨, 노랑
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255)) # 놀람, Cyan
        elif emotion_text == 'disgust':
            color = emotion_probability * np.asarray((178, 102, 255)) # 역겨움, 연보라
        elif emotion_text == 'fear':
            color = emotion_probability * np.asarray((255, 0, 255)) # 두려움, 진분홍
        elif emotion_text == 'neutral':
            color = emotion_probability * np.asarray((0, 255, 0)) # 평온, 연두
        else:
            color = emotion_probability * np.asarray((255, 255, 255)) # 판단 불가, 흰색

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)


    # 'q' 입력 시, 창 닫기

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
