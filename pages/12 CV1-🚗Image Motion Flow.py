import streamlit as st
import streamlit as st

st.set_page_config(
    page_icon="📹"
)

st.write("""
# 📹 CV1 - Image Motion Flow

### Project Links  
🔗 **Site**: [https://cv1-image-motion-flow.onrender.com/](https://cv1-image-motion-flow.onrender.com/)  
💻 **GitHub**: [CV1-Image-Motion-Flow(private)](https://github.com/weiz-me/CV1-Image-Motion-Flow)
▶️ **Demo Video**: [https://youtu.be/gW6Wj-rWLRo](https://youtu.be/gW6Wj-rWLRo)  

### Project Description:  
This project implements optical flow estimation using the **Lucas-Kanade method**, a foundational technique in computer vision. The system takes in two grayscale image frames and calculates the apparent motion (flow vectors) at each pixel based on temporal intensity changes. 

Built in **Python** with **NumPy**, **SciPy**, and **Matplotlib**, this project demonstrates key concepts in motion analysis, including image gradients and local motion estimation using a sliding window approach.

### Code Summary:
- **Optical Flow Algorithm**: Uses spatial and temporal image gradients with a window-based least squares solution.
- **Libraries**: NumPy and SciPy for numerical operations; Matplotlib for flow visualization.
- **Core Functions**:
  - `computeFlow(img1, img2, window_size)`: Applies Lucas-Kanade method to compute motion vectors `(u, v)` for each pixel.
  - `s(x)`: Helper function to visualize intermediate matrices and results.

### Features:
- Convolution-based image gradient estimation for performance.
- Handles edge cases such as singular matrices using exception handling.
- Demonstrates motion between frames in visual plots.

### Demo Video:
""")

st.video(data="https://youtu.be/gW6Wj-rWLRo")

st.write("""
### Demo Website (3 min to boot up the website):  
#### https://cv1-image-motion-flow.onrender.com/  
""")

st.components.v1.iframe("https://cv1-image-motion-flow.onrender.com/", width=1200, height=800, scrolling=True)


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


 




