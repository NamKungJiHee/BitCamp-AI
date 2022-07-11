# https://www.geeksforgeeks.org/rotate-a-picture-using-ndimage-rotate-scipy/
# https://pythonguides.com/scipy-ndimage-rotate/
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
import numpy as np 
import os
# The SciPy “ndimage” submodule is dedicated to image processing. Here, “ndimage” means an n-dimensional image.

#path = 'C:\\Users\\Lenovo\\Desktop\\Test\\easyocr\\ndimage.rotate\\picture\\real_image\\'
path = 'C:\\Users\\Lenovo\\Desktop\\Test\\easyocr\\ndimage.rotate\\gum.jpg'

img = image.imread(os.path.abspath('gum.jpg'))
#plt.imshow(img)
#plt.show()
# print(img)

# 조건문으로 각도당 이미지 출력되게끔 하기 #

#################################################################
# angle = 0
# output = ndimage.rotate(img, angle, reshape=True, mode = 'constant')
#output = ndimage.rotate(img, 45, axes=(1,1), reshape=True, mode = 'constant')

''' 
input(array_data): It is the input array or image that we want to rotate.
angle(float): It is used to specify the rotation angle like 20 degrees or 90 degrees.
axes(two ints in tuples): The plane of rotation is defined by the two axes.
output(datatype or array): It defines the datatype of the returned array, by default returned array datatype is the same as the input array.
mode: It is used to specify the modes ‘constant’, ‘mirror’, ‘wrap’, ‘reflect’, ‘nearest’.
'''
# plt.imshow(output)
# plt.show()

output = []
angle = 0 
for idx in range(36):  # 36
    angle += 10
    result = ndimage.rotate(img, angle, reshape=True, mode='constant')

    zfilNum=str(idx).zfill(3)
    cv2.imwrite(f'{"gum"+zfilNum+".jpg"}',result)  # 출력이미지 저장하기  imwrite(저장이름, 저장파일)
    #output.append(result)




# rows = 6
# columns = 6

# for img in output : 
#     image_index = i + 1 # image index 
#     ttitle = "Image{}".format(image_index ) # image title
#     plt.subplot(rows, columns, image_index) # subplot 
#     plt.title(ttitle)    # title 
#     # // plt.axis('off')
#     plt.xticks([])  # x = None 
#     plt.yticks([])  # y = None
#     plt.imshow(image)
# plt.show()
# # plt.imshow(output)
# # plt.show()