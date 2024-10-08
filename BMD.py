# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:41:21 2024

@author: user
"""

import streamlit as st 
from PIL import Image

import numpy as np
import cv2
from tensorflow.keras.models import load_model

modelh = load_model('BMDH15.h5' ,compile = False)

modelt = load_model('BMDT2.h5' ,compile = False)


background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img5.pic.in.th/file/secure-sv1/smsk-1e26f337bb6ec6813.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)



#st.title("Samutsakhon Osteoporosis Screening tool(S.O.S)")
st.markdown("<h1 style='text-align: center; color: black ; font-size: 25px ;'>Samutsakhon Osteoporosis Screening tool(SOS)</h1>", unsafe_allow_html=True)
img_file = st.file_uploader("เปิดไฟล์ภาพ")

col1, col2 = st.columns([1,1]) 
#col3, col4 = st.columns([1,1]) 


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
        st.write("predict Osteoporosis")
    with col2:
        if  float(TscoreHip) < -1 : 
            st.write(":red[Abnormal]")  
        else:
            st.write(":green[Normal]")  
    
        

    
        


    
        




 


   




    
    


    

   
    





