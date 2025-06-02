import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("""
# AWS-💬CUThen Chatting APP
         
### Project Links  
🔗 **GitHub**: [CUThen](https://github.com/weiz-me/CUThen)

### Project Description:  
Built a cloud-native networking/chat application designed to connect Columbia University students with shared backgrounds and interests, promoting meaningful conversations and friendships.

### Code Summary:
- **Frontend**: HTML/JavaScript hosted on **AWS S3**, integrated with **API Gateway** and **Lambda Functions** for seamless and decoupled backend communication.
- **Backend**:  
  - **AWS Lambda** used for profile management, chat logic, and user matching algorithms  
  - Real-time features implemented for personalized user experience
- **Storage & Search**:  
  - **DynamoDB** for scalable storage of user/chat data  
  - **Opensearch** for fast and flexible querying
- **DevOps**:  
  - Automated CI/CD with **AWS CodePipeline** for testing and deployment  
  - Enables flexible updates and streamlined maintenance

### Demo Video:
""")

st.video(data="https://youtu.be/zvcYTh0VbhI")