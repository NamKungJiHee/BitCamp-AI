import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

img = np.full(shape=(512,512,3),fill_value=255,dtype=np.uint8)
img = Image.fromarray(img) #img배열을 PIL이 처리가능하게 변환

draw = ImageDraw.Draw(img)
font=ImageFont.truetype("fonts/gulim.ttc",20)
org=(50,100)
text="Hello OpenCV!(한글)"
draw.text(org,text,font=font,fill=(0,0,0)) #text를 출력
img = np.array(img) #다시 OpenCV가 처리가능하게 np 배열로 변환

#OpenCV기준으로 text크기를 구해 사각형 생성
font=cv2.FONT_HERSHEY_SIMPLEX
text="Hello OpenCV!(한글)"
size, BaseLine=cv2.getTextSize(text,font,1,2)
cv2.rectangle(img,org,(org[0]+size[0],org[1]+size[1]),(0,0,255))
cv2.circle(img,org,3,(255,0,255),2)

cv2.imshow("A",img)
cv2.waitKey()
cv2.destroyAllWindows()


