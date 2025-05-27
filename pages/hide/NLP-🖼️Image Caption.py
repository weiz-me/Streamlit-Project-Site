import streamlit as st
#from predict_page import show_predict_page
from explore_page import show_explore_page, show_predict_page


#page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))


show_explore_page()
show_predict_page()


#a=st.button("View data preparation, analysis, and model result")

