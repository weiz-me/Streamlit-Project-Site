import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("Site: https://vehicle-project.onrender.com/")

st.components.v1.iframe("https://vehicle-project.onrender.com/?embed=True", width=1500, height=1000, scrolling=True)

