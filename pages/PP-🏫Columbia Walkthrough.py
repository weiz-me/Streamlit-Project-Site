import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("Site: https://weiz-me.github.io/columbia-walkthrough/")

st.components.v1.iframe("https://weiz-me.github.io/columbia-walkthrough/?embed=True", width=1500, height=1000, scrolling=True)

