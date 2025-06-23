import streamlit as st

st.set_page_config(page_title="AML-movie Score", page_icon="🎬")

st.write("""
# AML-movie Score 🎬📊

### Project Links  
▶️ **Demo Website**: [https://ml-movie-uqee.onrender.com](https://ml-movie-uqee.onrender.com)  
▶️ **Demo Video**: [https://www.youtube.com/watch?v=UsmnAS9HM-0]https://www.youtube.com/watch?v=UsmnAS9HM-0)  
💻 **GitHub**: [AML-movie-score (private)](https://github.com/weiz-me/ML_Movie)

### Project Description
This project investigates the prediction of movie **popularity** and **average vote score** based on structured features using various machine learning techniques.

The application includes:
- 🧹 **Data Cleaning & Feature Engineering**: Handling missing data, encoding genres, directors, actors, and categorical sentiment scores (e.g., rhythm, effort)
- 📈 **Exploratory Data Analysis**: Distribution and correlation visualizations for numeric and categorical attributes
- 🧠 **Modeling**:
  - **Regression** (predicting average vote):
    - Linear Regression, Lasso, Ridge
  - **Classification** (predicting popularity label):
    - Logistic Regression, Decision Tree, Random Forest, Neural Network

### Dataset Info
- Source: Custom dataset of 8000+ films  
- Features: Genre, year, country, duration, directors, actors, ratings, viewer sentiments, etc.

### Project Demo Video
""")

# Embed demo video
st.video("https://www.youtube.com/watch?v=UsmnAS9HM-0")

st.write("### Project Site (3 min boot up): https://ml-movie-uqee.onrender.com/")

st.components.v1.iframe("https://ml-movie-uqee.onrender.com/", width=1500, height=1000, scrolling=True)

