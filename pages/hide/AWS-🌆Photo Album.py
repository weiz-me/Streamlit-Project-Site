import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("Site: https://b1-frontendwz.s3.amazonaws.com/index.html")

st.components.v1.iframe("https://b1-frontendwz.s3.amazonaws.com/index.html", width=1500, height=1000, scrolling=True)

