import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)

st.write("""
# AWS-📷Photo Album

### Project Links  
🔒 **Backend Repo**: [cs6998-hw2-submission (private)](https://github.com/weiz-me/cs6998-hw2-submission)  
🔒 **Frontend Repo**: [cs6998-hw2-frontend (private)](https://github.com/weiz-me/cs6998-hw2-frontend)

### Project Description:  
A cloud-native photo album application that supports image searching via **text and voice commands**. The platform utilizes **AWS Rekognition** for image labeling and **Amazon Lex** for conversational voice search, providing a highly interactive user experience.

### Key Features:
- **Image Indexing & Storage**:  
  - Users upload photos to an **S3 bucket**  
  - **AWS Rekognition** detects labels from images  
  - A **Lambda function** stores labels and metadata in **OpenSearch**
- **Search Functionality**:  
  - Users can search via **typed queries** or **voice input**  
  - Voice queries processed using **Amazon Transcribe**  
  - Search results retrieved via **API Gateway**
- **Frontend**:  
  - Built with HTML/JavaScript  
  - Hosted on **S3** for scalability and low latency  
  - Interface supports photo upload, custom labeling, and voice/text search
- **DevOps & Infrastructure**:  
  - Automated deployments using **AWS CodePipeline**  
  - Managed infrastructure via **AWS CloudFormation**
""")