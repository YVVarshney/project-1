# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:55:34 2018

@author: masters
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:18:08 2018

@author: masters
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


def folder_path(v):
    if v==1:
        folder_name='C:\\Users\\masters\\Desktop\\moving object detection\\keyring'
    if v==2:
        folder_name='C:\\Users\\masters\\Desktop\\moving object detection\\mobile'
    if v==3:
        folder_name='C:\\Users\\masters\\Desktop\\moving object detection\\mouse'
    if v==4:
        folder_name='C:\\Users\\masters\\Desktop\\moving object detection\\stappler'
    if v==5:
        folder_name='C:\\Users\\masters\\Desktop\\moving object detection\\all data'    
    return folder_name

def diffImg(t0, t1):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t1, t0)
  d1 = cv2.medianBlur(d1,3)  
  d1 = cv2.bilateralFilter(d1,3,75,75)
  d1[d1<30]=0
  return d1

def img_class(x):
    return {
         0:'keyring', 1:'mobile', 2:'mouse', 3:'stappler',
    }[x]
#def img_class(x):
#    return {
#         0:'bag', 1:'car', 2:'car', 3:'car', 4:'car', 5:'Legs', 6:'MrCA1', 7:'MrCA1',
#         8:'MrCA2', 9:'MrCA2back', 10:'MrCL2', 11:'MrMB1', 12:'MrMB1down', 13:'MrMB2', 
#         14:'MrMB2 with bag', 15:'MrOC1', 16:'MrRA1', 17:'MrRA21', 18:'MrRA22', 19:'MrSU1',
#         20:'MsCL2', 21:'MsOC2', 22:'MsSI2', 23:'Shadow', 24:'SI1_1', 25:'SI1_2', 26:'SI1_3', 
#         27:'object missed',
#    }[x]
    
def print_text(cls_cnt,blank):
    text1="Keyring: "+str(cls_cnt[0])    
    text2="Mobile: "+str(cls_cnt[1])
    text3="Mouse: "+str(cls_cnt[2])
    text4="Stappler: "+str(cls_cnt[3])    
    cv2.putText(blank, text1, (0,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text2, (0,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text3, (0,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text4, (0,110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    return blank