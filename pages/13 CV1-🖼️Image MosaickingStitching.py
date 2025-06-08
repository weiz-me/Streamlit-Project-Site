import streamlit as st

st.set_page_config(
    page_icon="🧵"
)

st.write("""
# 🧵 CV1 - Image Mosaicking & Stitching

### Project Links  
▶️ **Demo Video**: [https://youtu.be/5jlrjD9x6Bc](https://youtu.be/5jlrjD9x6Bc)  
💻 **GitHub**: [CV1-Image-MosaickingStitching(private)](https://github.com/weiz-me/CV1-Image-MosaickingStitching)

### Project Description:  
This project implements a complete **image mosaicking system** from scratch using **homography estimation**, **RANSAC**, and **backward warping**. Given a set of overlapping images, the algorithm computes keypoint correspondences, estimates the transformation, and blends them into a seamless panorama. It's part of a CV1 assignment focused on geometric vision techniques.

### Code Summary:
- **Homography Estimation**: Uses least-squares via eigenvector decomposition to compute 3×3 transformations.
- **RANSAC**: Identifies inliers and filters out noisy correspondences between image pairs.
- **Backward Warping**: Uses `cv2.remap` for pixel-accurate image projection using inverse homography.
- **Blending**: Supports both "overlay" and smooth "distance-based blend" modes for stitching.
- **Utility**: Modular functions allow stitching of arbitrary image sequences and panoramas.

### Highlights:
- Built using **NumPy**, **OpenCV**, and **SciPy**.
- SIFT-based feature matching with custom stitching logic.
- Visualizations and mosaic generation via Matplotlib and utility helpers.

### Demo Video:
Watch the live demo here:  
📺 [https://youtu.be/5jlrjD9x6Bc](https://youtu.be/5jlrjD9x6Bc)
""")

# You can embed video with st.video or just use the link as above.
st.video("https://youtu.be/5jlrjD9x6Bc")

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


 




