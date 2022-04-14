import cv2
import mediapipe as mp
import playsound

import google_tts

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 

# 얼굴 탐지 모델 가중치
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
	'deploy_age.prototxt',
	'age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
	'deploy_gender.prototxt',
	'gender_net.caffemodel')

age_list = ['0,2','4,6','8,12','15,20','25,32','38,43','48,53','60,100']
gender_list = ['남자', '여자'] 

def pic_detection(image):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        # 작업 전에 BGR 이미지를 RGB로 변환합니다.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 이미지를 출력하고 그 위에 얼굴 박스를 그립니다.
        if not results.detections:
            print(msg)
            msg = "카메라에 얼굴이 찍히지 않았습니다"
            print(msg)
            google_tts.synthesize_text(msg)
            playsound.playsound("output.mp3")
            return
            
        annotated_image = image.copy()
        blob = cv2.dnn.blobFromImage(annotated_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_preds.argmax()
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds.argmax()
        
        age_1 = age_list[age].split(',')[0]
        age_2 = age_list[age].split(',')[1]
        
        msg = f'당신은 {age_1} 세 에서 {age_2}세 사이의 {gender_list[gender]} 입니다'
        print(msg)
        google_tts.synthesize_text(msg) # synthesize_text: text를 읽어라
        playsound.playsound("output.mp3")

        # 사진 출력
    cv2.imshow('facenet',annotated_image)  
    cv2.waitKey(10000)

if __name__ =='__main__':
    import selfy
    selfy.selfy()
    playsound.playsound("camera.mp3")
    img = cv2.imread('my_pic.jpg') 
    pic_detection(img)

