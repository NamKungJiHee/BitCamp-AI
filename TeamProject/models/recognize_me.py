# -*- coding: UTF-8 -*-

import sys,os,dlib,glob,numpy
from skimage import io
import cv2

#predictor_path="shape_predictor_68_face_landmarks.dat"
# 1.       

detector = dlib.get_frontal_face_detector()

# 2.          
# sp = dlib.shape_predictor(predictor_path)
path = 'C:\\Users\\bitcamp\\Desktop\\TP\\test_img\\'
img = io.imread(path + "test.jpg")
# 1.    
dets = detector(img, 3)
print("Number of faces detected: {}".format(len(dets)))

"""
win = dlib.image_window()

for k, d in enumerate(dets):
    # 2.     
    shape = sp(img, d)
    #            
    win.clear_overlay()
    win.add_overlay(d)
    win.add_overlay(shape)
"""
for d in dets:
    # print(d)
    # print(type(d))
    #   opencv          
    left_top=(dlib.rectangle.left(d),dlib.rectangle.top(d))
    right_bottom=(dlib.rectangle.right(d),dlib.rectangle.bottom(d))
    cv2.rectangle(img,left_top,right_bottom,(0,255,0),2,cv2.LINE_AA)

cv2.imshow("img",cv2.cvtColor(img,cv2.COLOR_RGB2BGR)) #   ＢＧＲ  
cv2.waitKey(0)
cv2.destroyAllWindows()