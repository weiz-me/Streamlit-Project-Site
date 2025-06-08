import streamlit as st

st.set_page_config(
    page_icon="🧔"
)

st.write("""
# 🧔 Beard Detection with Deep Learning

### Project Links  
💻 **GitHub**: [Beard-Detection(private)](https://github.com/weiz-me/Beard-Detection)
▶️ **Demo Video**: [https://youtu.be/xB3ublOfCqg](https://youtu.be/xB3ublOfCqg)  

### Project Description:  
A deep learning-based web app that detects whether a person has a beard in an image. Developed using **Python** and **TensorFlow**, the model uses a Convolutional Neural Network (CNN) trained on facial images to classify "Beard" or "No Beard." The web UI is powered by **Streamlit** for real-time image upload and inference.

### Code Summary:
- **Model**: Custom CNN trained using TensorFlow/Keras on a labeled beard image dataset.
- **Frontend**: Built with Streamlit for easy-to-use drag-and-drop interface.
         
### Demo Video:
""")

st.video(data="https://youtu.be/xB3ublOfCqg")
