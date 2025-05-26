import streamlit as st
import sys

import cv2
import numpy as np
import solutions_wz2580_hw7 as solutions
import utils_hw7 as utils


def upload_to_np(uploaded_file):

    # Decode the image using OpenCV
   # image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

   img1 = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

   # Convert dtype to np.float64
   # img1_float64 = (image/255).astype(np.float64)

   return img1

st.title("CV1-Image Motion Flow")
st.write("#### Input your own images")
col1, col2 = st.columns(2)
col11, col22 = st.columns(2)

with col1:
   left_image = st.file_uploader("Upload your base image.")
   if left_image:
      l = upload_to_np(left_image)
      st.image(left_image)
      st.image(l)

with col2:
   middle_image = st.file_uploader("Upload your next image.")
   if middle_image:
      st.image(middle_image)
      m = upload_to_np(middle_image)
      st.image(m)
with col11:
   ws = st.slider("window size",1,100,20)
with col22:
   scale = st.slider("scale",1,30,15)
stitch = st.button("Image Motion")
if stitch:
   if not left_image:
      st.write("please input first image..")
   elif not middle_image:
      st.write("please input next image")
   else:
      img1_gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
      img2_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
      flow = solutions.computeFlow(img1_gray, img2_gray, window_size=ws)
      needle = utils.draw_flow_arrows(img1_gray, flow, step=16, scale=scale, L=4)
      st.image(needle)


st.write("#### Input Example")
col1, col2 = st.columns(2)
with col1:
   st.write("first image")
   st.image("frame_18_delay-0.1s.jpg")

with col2:
   st.write("second image")
   st.image("frame_19_delay-0.1s.jpg")

st.write("#### Result")
st.image("r1.jpg")


 




