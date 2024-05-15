# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:41:21 2024

@author: user
"""

import streamlit as st 
from PIL import Image
#import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

modelh = load_model('BMDH11.h5' ,compile = False)

modelt = load_model('BMDT2.h5' ,compile = False)

#image_path = 'C:/Users/user/Desktop/วิจัย BMD/9286C.jpg'

image_height = 128
image_width = 128


def Bone(a):
   if float(a) < -2.5 :
       return 'Osteoporosis'
   if float(a) < -1:
       return 'Osteopenia'
   else:
       return 'Normal'




st.title("Predict BMD")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

col1, col2 = st.columns([1,1]) 
col3, col4 = st.columns([1,1]) 


if img_file is not None:
   
    im = Image.open(img_file)

    st.image(img_file,channels="BGR")
    
    img= np.asarray(im).astype(np.float32) /255.0 
    image= cv2.resize(img,(128, 128))
    X_submission = np.array(image)
    y = np.expand_dims(X_submission, 0)
    
    result1 = modelh.predict(y)
    
    result2 = modelt.predict(y)
    
    TscoreSpine = f'{result2[0][0]:.1f}' 
    TscoreHip= f'{result1[0][0]:.1f}'
    

    with col1:
        st.write("predict T-score spine" )
    with col2:
        st.code(f"""{TscoreSpine}  {Bone(TscoreSpine)}""") 
        

    

    with col3:
        st.write("predict T-score hip")
    with col4:
        st.code(f"""{TscoreHip}  {Bone(TscoreHip)}""")  
        




 


   




    
    


    

   
    





