import streamlit as st
import sys

import cv2
import numpy as np
import solutions_wz2580_hw4 as solutions
import utils_hw4 as utils


def upload_to_np(uploaded_file):

    # Decode the image using OpenCV
   image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), 1)

   # img1 = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), cv2.IMREAD_ANYCOLOR)

   # Convert dtype to np.float64
   img1_float64 = (image/255).astype(np.float64)

   return img1_float64

st.title("CV1-Image MosaickingStitching")
st.write("#### Input your own images")
col1, col2, col3 = st.columns(3)

with col1:
   left_image = st.file_uploader("Upload your left image.")
   if left_image:
      l = upload_to_np(left_image)
      st.image(left_image)
      st.image(l)

with col2:
   middle_image = st.file_uploader("Upload your middle image (base image).")
   if middle_image:
      st.image(middle_image)
      m = upload_to_np(middle_image)
      st.image(m)


with col3:
   right_image = st.file_uploader("Upload your right image (optional).")
   if right_image:
      st.image(right_image)
      r = upload_to_np(right_image)
      st.image(r)

stitch = st.button("stitch images")
if stitch:
    if not left_image:
      st.write("please input left image..")
    elif not middle_image:
      st.write("please input middle image")
    elif not right_image:
      img = solutions.stitch_imgs([m,l])
    else:
      img = solutions.stitch_imgs([m,l,r])


    img = np.clip(img, 0, 1)
    st.image(img, caption="result", use_column_width=True)
st.write("#### Input Example")
col1, col2, col3 = st.columns(3)
with col1:
   st.write("left image")
   st.image("room-left.jpg")

with col2:
   st.write("middle image")
   st.image("room-center.jpg")

with col3:
   st.write("right image")
   st.image("room-right.jpg")

st.write("#### Result")
st.image("1f.png")


 




