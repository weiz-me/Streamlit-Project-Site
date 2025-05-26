import streamlit as st

st.title("PP - Mask and Beard Detection")

st.write("#### Input your own images")
uploaded_file = st.file_uploader("Upload your image (jpg, jpeg, png)..")

camera_img = st.camera_input("Take a photo")

import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import PIL
import numpy as np
import keras.utils as image


mask_model = tf.keras.models.load_model('mask.h5')
beard_model = tf.keras.models.load_model('beard.h5')

if uploaded_file:
    test_image = image.load_img(uploaded_file, target_size = (256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result_mask = mask_model.predict(test_image)
    if result_mask[0][0] > 0.5:
        prediction1 = 'no'
    else:
        prediction1 = 'yes'
    
    result_beard = beard_model.predict(test_image)
    if result_beard[0][0] > 0.5:
        prediction2 = "no"
    else:
        prediction2 = "yes"


    st.image(uploaded_file)
    st.write(f"mask: {prediction1}")
    st.write(f"beard: {prediction2}")


if camera_img:
    test_image = image.load_img(camera_img, target_size = (256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result_mask = mask_model.predict(test_image)
    if result_mask[0][0] > 0.5:
        prediction1 = 'no'
    else:
        prediction1 = 'yes'
    
    result_beard = beard_model.predict(test_image)
    if result_beard[0][0] > 0.5:
        prediction2 = "no"
    else:
        prediction2 = "yes"


    st.image(camera_img)
    st.write(f"mask: {prediction1}")
    st.write(f"beard: {prediction2}")

# if camera_img:
#     st.image(camera_img)
#from predict_page import show_predict_page


#page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))


# show_explore_page()
# show_predict_page()


#a=st.button("View data preparation, analysis, and model result")

