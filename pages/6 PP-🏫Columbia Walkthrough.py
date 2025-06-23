import streamlit as st

st.write("""
# Columbia Walkthrough Project 🎓📍

### Project Links  
▶️ **Live Tour**: [https://weiz-me.github.io/columbia-walkthrough](https://weiz-me.github.io/columbia-walkthrough)  
▶️ **Demo Video**: [https://youtu.be/iGKqBbKIkoE](https://youtu.be/iGKqBbKIkoE)  
💻 **GitHub**: [Columbia-Walkthrough (public)](https://github.com/weiz-me/columbia-walkthrough)

### Project Description
This web-based interactive visualization provides a virtual walkthrough of **Columbia University’s Morningside Campus** using:
- 🗺️ **Leaflet Maps** to plot building locations
- 🔄 **Panolens.js + Three.js** for immersive 360° panoramas
- 📸 Over **85 panoramic views** stitched and linked across clickable markers

Key Features:
- Clickable map markers that launch immersive 360° panoramas
- Dynamic transitions and location syncing between map and view
- Responsive HTML layout with full-screen visualization capabilities

### Technologies Used:
- **HTML5**, **JavaScript**, **Panolens.js**, **Leaflet.js**
- Hosted with **Render.com**

### Demo Video
""")

# Embed YouTube video placeholder
st.video("https://www.youtube.com/watch?v=your_demo_video_id")

st.write("### Project Site: https://weiz-me.github.io/columbia-walkthrough/")

st.components.v1.iframe("https://weiz-me.github.io/columbia-walkthrough/?embed=True", width=1500, height=1000, scrolling=True)

