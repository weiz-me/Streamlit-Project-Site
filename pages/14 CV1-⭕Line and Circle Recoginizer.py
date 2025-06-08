import streamlit as st

st.set_page_config(page_title="CV1 - Line and Circle Recognizer", page_icon="⭕")

st.write("""
# CV1 - Line and Circle Recognizer

### Project Links  
▶️ **Demo Website**:[https://cv1-line-and-circle-recoginizer.onrender.com](https://cv1-line-and-circle-recoginizer.onrender.com)
         
▶️ **Demo Video**: [https://youtu.be/rEAayDUxLrQ](https://youtu.be/rEAayDUxLrQ)  
💻 **GitHub**: [CV1-Line-and-Circle-Recoginizer(private)](https://github.com/weiz-me/CV1-Line-and-Circle-Recoginizer)

### Project Description
This application demonstrates computer vision techniques for:
- Detecting **lines** using Canny edge detection and the Hough transform
- Detecting **circles** (e.g., coins) via a custom implementation of the Circle Hough Transform

The code leverages libraries such as `OpenCV`, `NumPy`, `matplotlib`, and `skimage`.

### Project Demo Video
""")

st.video("https://www.youtube.com/watch?v=rEAayDUxLrQ")

st.write("### Project Site (3 min bootup): https://cv1-line-and-circle-recoginizer.onrender.com")

st.components.v1.iframe("https://cv1-line-and-circle-recoginizer.onrender.com", width=1500, height=1000, scrolling=True)



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
