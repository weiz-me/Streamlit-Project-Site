import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("""
# ASE-🤖Prompt Service API and Client App

### Project Links  
🔗 **API Repo**: [ASE4156](https://github.com/ASE4156/ASE4156)  
💻 **Client App Repo**: [client-app](https://github.com/ASE4156/client-app)
         
### Project Description:
Designed and implemented a robust **RESTful API service in C++** for managing user data, authentication tokens, and ChatGPT prompt sessions within a **PostgreSQL** database. The service allows clients to create and persist multiple ChatGPT conversations securely.

### Key Features:
- **C++ Backend**: Custom-built REST API to handle user and session data efficiently.
- **Database Integration**: PostgreSQL used for storing user information and prompt histories.
- **Testing & Quality Assurance**:  
  - Unit and end-to-end tests with **Catch2**  
  - Code style checks using **cpplint**  
  - Static code analysis via **cppcheck**  
  - Branch coverage tests to ensure full code reliability
### Demo Video:
""")

st.video(data="https://www.youtube.com/watch?v=tAgX81AnuOI")