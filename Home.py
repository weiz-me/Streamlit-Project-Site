import streamlit as st


st.set_page_config(
    page_title="Wei's Projects Site",
    page_icon="🧑‍💻"
)

st.write("# Welcome to Wei's Project Site! 👋")
st.sidebar.success("Select a Project above.")

st.markdown(
    """

    **👈 Select a Project from the dropdown on the left**

    ### Project List
    Hi, I am currently looking for internship and full time.

    - HPML: Trasnformer Inference Optimization (https://github.com/weiz-me/hpml_proj)
    - ASE: Prompt Service API and Client App (https://github.com/orgs/ASE4156/repositories)
    - AWS: CUThen Chatting APP - Fall 2023 (https://github.com/Infernaught/CUThen)
    - AWS: Photo Album - Fall 2023
    - CV1: Image Motion Flow - Fall 2023
    - CV1: Image Mosaicking/Stitching - Fall 2023
    - CV1: Line and Circle Recoginizer - Fall 2023
    - NLP: Image Caption Generator - Spring 2023
    - AML: Movie Scoring Prediction - Spring 2023
    - DB: Vehicle Database - Spring 2023
    - AI: Tic-Tac-Toe AI - Spring 2023
    - PP: Columbia Walkthrough - March 2023


    

    
    ##### Acronym
        AI: Artificial Intelligence, 
        AML: Applied Machine Learning, 
        ASE: Advanced Software Engineering, 
        AWS: Amazon Web Services, 
        CV1: Computer Vision one, 
        DB: Database, 
        ML: Machine Learning, 
        NLP: Natural Language Processing, 
        PP: Personal Project
        HPML: High Performance Machine Learning
    """
)