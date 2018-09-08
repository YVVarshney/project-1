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


def load_images_from_folder(v):
    if v==1:
        folder_name='D:\\CV dataaset LASIESTA\\I_CA_01\\I_CA_01'
        limit=350
        file_first='I_CA_01'
    if v==2:
        folder_name='D:\\CV dataaset LASIESTA\\I_CA_02\\I_CA_02'
        limit=525
        file_first='I_CA_02'
    if v==3:
        folder_name='D:\\CV dataaset LASIESTA\\I_MB_01\\I_MB_01'
        limit=275
        file_first='I_MB_01'
    if v==4:
        folder_name='D:\\CV dataaset LASIESTA\\I_MB_02\\I_MB_02'
        limit=350
        file_first='I_MB_02'
    if v==5:
        folder_name='D:\\CV dataaset LASIESTA\\I_OC_01\\I_OC_01'
        limit=250
        file_first='I_OC_01'
    if v==6:
        folder_name='D:\\CV dataaset LASIESTA\\I_OC_02\\I_OC_02'
        limit=250
        file_first='I_OC_02'
    if v==7:
        folder_name='D:\\CV dataaset LASIESTA\\I_SI_01\\I_SI_01'
        limit=300
        file_first='I_SI_01'
    if v==8:
        folder_name='D:\\CV dataaset LASIESTA\\I_SI_02\\I_SI_02'
        limit=300
        file_first='I_SI_02'
    if v==9:
        folder_name='D:\\CV dataaset LASIESTA\\O_CL_01\\O_CL_01'
        limit=225
        file_first='O_CL_01'
    if v==10:
        folder_name='D:\\CV dataaset LASIESTA\\O_CL_02\\O_CL_02'
        limit=425
        file_first='O_CL_02'
    if v==11:
        folder_name='D:\\CV dataaset LASIESTA\\O_RA_01\\O_RA_01'
        limit=1400
        file_first='O_RA_01'
    if v==12:
        folder_name='D:\\CV dataaset LASIESTA\\O_RA_02\\O_RA_02'
        limit=375
        file_first='O_RA_02'
    if v==13:
        folder_name='D:\\CV dataaset LASIESTA\\O_SN_01\\O_SN_01'
        limit=500
        file_first='O_SN_01'
    if v==14:
        folder_name='D:\\CV dataaset LASIESTA\\O_SN_02\\O_SN_02'
        limit=850  
        file_first='O_SN_02'
    images = []
    for i in range(limit):
        img = cv2.imread(folder_name+'\\'+file_first+'-'+str(i+1)+'.bmp')
        if img is not None:
            images.append(img)
    return images, folder_name

def diffImg(t0, t1):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t1, t0)
  d1 = cv2.medianBlur(d1,3)  
  d1 = cv2.bilateralFilter(d1,3,75,75)
  d1[d1<30]=0
  return d1

def img_class(x):
    return {
         0:'bag', 1:'car', 2:'car', 3:'car', 4:'car', 5:'Legs', 6:'MrCA1', 7:'MrCA1',
         8:'MrCA2', 9:'MrCA2', 10:'MrCL2', 11:'MrMB1', 12:'MrMB1', 13:'MrMB2', 
         14:'MrMB2', 15:'MrOC1', 16:'MrRA1', 17:'MrRA21', 18:'MrRA22', 19:'MrSU1',
         20:'MsCL2', 21:'MsOC2', 22:'MsSI2', 23:'Shadow', 24:'SI1_1', 25:'SI1_2', 26:'SI1_3', 
         27:'object missed',
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
    text1="Bag: "+str(cls_cnt[0])    
    text2="Car: "+str(cls_cnt[1]+cls_cnt[2]+cls_cnt[3]+cls_cnt[4])
    text3="Legs: "+str(cls_cnt[5])
    text4="CA1: "+str(cls_cnt[6]+cls_cnt[7])    
    text5="CA2: "+str(cls_cnt[8]+cls_cnt[9])  
    text6="CL2: "+str(cls_cnt[10])   
    text7="MB1: "+str(cls_cnt[11]+cls_cnt[12])
    text8="MB2: "+str(cls_cnt[13]+cls_cnt[14])
    text10="OC1: "+str(cls_cnt[15])
    text11="RA1: "+str(cls_cnt[16])   
    text12="RA21: "+str(cls_cnt[17])
    text13="RA22: "+str(cls_cnt[18])   
    text14="SU1: "+str(cls_cnt[19])
    text15="CL2: "+str(cls_cnt[20])   
    text16="OC2: "+str(cls_cnt[21])
    text17="SI2: "+str(cls_cnt[22])   
    text18="Shadow: "+str(cls_cnt[23])
    text19="SI1_1: "+str(cls_cnt[24])   
    text20="SI1_2: "+str(cls_cnt[25])
    text21="SI1_3: "+str(cls_cnt[26])   
    text22="Missed: "+str(cls_cnt[27])
    cv2.putText(blank, text1, (0,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text2, (0,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text3, (0,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text4, (0,110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text5, (0,140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text6, (0,170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text7, (0,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text8, (120,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text10, (120,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text11, (120,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text12, (120,110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text13, (120,140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text14, (120,170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text15, (120,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text16, (240,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text17, (240,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text18, (240,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text19, (240,110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text20, (240,140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text21, (240,170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    cv2.putText(blank, text22, (240,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
    return blank