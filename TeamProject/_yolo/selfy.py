import numpy as np
import cv2

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용 # 입력부분을 관리하는 함수
cap.set(3,640) # 너비# cap.set()함수로 설정하면 새로운 폭과 높이를 설정할 수 있음
cap.set(4,480) # 높이

def selfy():
    ret, frame = cap.read() # 사진 촬영# cap.read()는 재생되는 비디오의 한 프레임씩 읽는다
    frame = cv2.flip(frame, 1) # 좌우 대칭
    # ret = 비디오 프레임을 제대로 읽었는가? / frame = 읽은 프레임
    cv2.imwrite('my_pic.jpg', frame) # 사진 저장
    # imwrite() 함수: 이미지 저장 
    cap.release()
    cv2.destroyAllWindows()# 생성한 모든 윈도우를 제거
    print("찰칵~")
    
if __name__ =='__main__':
    selfy()
    
    