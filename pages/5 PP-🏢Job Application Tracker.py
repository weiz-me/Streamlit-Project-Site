import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)

st.write("""
# PP-🏢Job Application Tracker

### Project Links  
🔗 **Site**: [Project site](https://tracking-job-ap-frontend.vercel.app)  
💻 **GitHub**: [Tracking_Job_AP](https://github.com/weiz-me/Tracking_Job_AP)

### Project Description:  
A full-stack web application designed to help users track and manage job applications. Built using **React** for the frontend, **Node.js (Express)** for the backend, and **PostgreSQL** for data storage. The platform includes secure user authentication with **JWT** and **bcrypt**, role-based access for users and admins, and a scraping service that auto-collects job listings from external sources and stores them with validation.

### Code Summary:
- **Frontend**: Multi-page UI built with React and React Router for smooth navigation.
- **Backend**: Express.js RESTful API with secure routes, JWT-based auth, and bcrypt password hashing.
- **Database**: PostgreSQL schema for users, job entries, and admin control.
- **Scraping Service**: OpenAI API to fetch and parse job data from target websites.
         
### Demo Video:
         """)

st.video(data="https://youtu.be/aANh3Hn4ciM")

st.write("""
### Demo Website: 
#### Frontend: https://tracking-job-ap-frontend.vercel.app
""")
st.components.v1.iframe("https://tracking-job-ap-frontend.vercel.app", width=1200, height=800, scrolling=True)


st.write("""
#### Backend: https://tracking-job-ap-backend.onrender.com/
""")
st.components.v1.iframe("https://tracking-job-ap-backend.onrender.com/", width=1200, height=800, scrolling=True)
