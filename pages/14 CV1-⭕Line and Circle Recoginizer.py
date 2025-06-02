import streamlit as st
import sys

import cv2
import numpy as np
import solutions_wz2580_hw3 as solutions
import utils_hw3 as utils
st.title("CV1-Line and Circle Recoginizer")

st.write("#### Input your own images")
uploaded_file = st.file_uploader("Upload your image (jpg, jpeg, png)..")
col1, col2= st.columns(2)

if uploaded_file:
    # image = PIL.Image.open(uploaded_file)
    image_bytes = uploaded_file.read()

    # Convert the bytes to a numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image using OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Display the image
    with col1:
        st.write("## Orginal images")
        st.image(img, channels="BGR")
    with col2:
        edge_img = solutions.find_edge_pixels(img)
        st.write("## Edge images")
        st.image(edge_img)
    thrshold = st.slider("Increase value for fewer lines", min_value=30, max_value=300, value=110)
    line = st.button("Recoginize Lines")

    if line:
        hough_accumulator = solutions.generate_hough_accumulator(edge_img)
        line_img = solutions.line_finder(img, hough_accumulator,thrshold)
        st.image(line_img)

    col11, col22, col33= st.columns(3)
    with col11:
        minrad = st.slider("min radius", min_value=1, max_value=100, value=50)
    with col22:
        maxrad = st.slider("max radius", min_value=1, max_value=500, value=300)
    with col33:
        mindist = st.slider("min distance between circles", min_value=1, max_value=500, value=300)
    circle = st.button("Recoginize Circles")
    if circle:
        # gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        binary = utils.binarize(edge_img)
        blurred = cv2.GaussianBlur(edge_img, (3, 3), 0)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, dp=1, minDist=mindist,
            param1=30, param2=20, minRadius=minrad, maxRadius=maxrad
        )
        if circles is not None:
            count = 0
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 10)
                count+=1
                # draw the center of the circle
                # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.putText(img, str(count), (i[0] - 10, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        circle_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(circle_img)
        st.write(f"Num of Circle: {count}")



    #     num_labels, labeled_img = cv2.connectedComponents(binary_img)

    #     st.image(binary_img)
        # st.image(labeled_img)


st.write("#### Line Example")
col1a, col2a= st.columns(2)
with col1a:
    st.image("line_sample1o.jpg")
with col2a:
    st.image("line_sample1.jpg")
st.write("#### Circle Example")
col1b, col2b= st.columns(2)
with col1b:
    st.image("circle_sample1.jpg")
with col2b:
    st.image("circle_sample1o.jpg")



 




