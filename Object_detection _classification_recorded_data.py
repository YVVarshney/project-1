# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:49:59 2018

@author: Yash Vardhan Varshney
"""

import numpy as np		      # importing Numpy for use w/ OpenCV
import cv2                            # importing Python OpenCV
import os
from keras.models import  load_model
from datetime import datetime         # importing datetime for naming files w/ timestamp
#from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
font=cv2.FONT_HERSHEY_SIMPLEX
from support_project_objects import folder_path, diffImg, img_class, print_text

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


#cam = cv2.VideoCapture(0)             # Lets initialize capture on webcam
print ('1. keyring, 2. mobile, 3. mouse, 4. stappler, 5. all data')
print('Choose any video file')
v=int(input())

folder=folder_path(v)
images=load_images_from_folder(folder)
model=load_model('yash_object_classifier_new_objects.h5')
# Read three images first:
blank_img=np.zeros(images[0].shape,dtype=np.uint8)
blank_img[:]=255
t_background = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY)
timeCheck = datetime.now().strftime('%Ss')
i=0
winName = "Movement Indicator"	      # comment to hide window
cv2.namedWindow(winName)              # comment to hide window
class_count=np.zeros(28)
img_last_class=['nothing']
image_class=['a']
for filename in os.listdir(folder):
  img_or=images[i]
  t = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
 # ret, frame = cam.read()	      # read from camera
  img_diff=diffImg(t_background, t)
  ret,thresh = cv2.threshold(img_diff,50,255,cv2.THRESH_BINARY)
  kernel = np.ones((40,20), np.uint8)
  img_dilation = cv2.dilate(thresh, kernel, iterations=1)
  im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
  if sorted_ctrs ==[]:
      img_last_class=['nothing']
  for j, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    if w > 35 and h > 40 and j<2:
#            img = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)
            img=img_or[y:y+h,x:x+w,:]
            img=cv2.resize(img,(100,250))
            img = img.reshape(1,100,250,3)
            classes=model.predict(img)
            class_index=np.where(classes>0.5)[1][0]
            image_class=img_class(class_index)
            if image_class!=img_last_class:
                class_count[class_index]=class_count[class_index]+1
           # if sum(sum(sum(img_or[y:y+h,x:x+w,:]-cv2.resize(img_last, (w, h)) )))>100:
            cv2.rectangle(img_or,(x,y),( x + w, y + h ),(0,255,0),2)
            cv2.putText(img_or,image_class,(x,y+20), font, 0.8, (0,255,0),2,cv2.LINE_AA)
            img_last_class=image_class
#  images2=np.hstack((img_or,cv2.cvtColor(img_diff, cv2.COLOR_GRAY2BGR)))
  images2=np.hstack((img_or,print_text(class_count.astype(int),np.copy(blank_img))))
  cv2.imshow(winName,images2)
  i=i+1
  

  key = cv2.waitKey(1) & 0xFF
  if key == 27:			 # comment this 'if' to hide window
    cv2.destroyAllWindow(winName)
    break

cv2.waitKey(0)