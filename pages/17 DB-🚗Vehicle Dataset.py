import streamlit as st

st.set_page_config(page_title="DB vehicle database", page_icon="🚗")

st.write("""
# DB Vehicle Database 🚘🗃️

### Project Links  
▶️ **Demo Website**: [https://vehicle-project.onrender.com/](https://vehicle-project.onrender.com/)  
▶️ **Demo Video**: [https://www.youtube.com/watch?v=gWhkikwFfBM&t=2s](https://www.youtube.com/watch?v=gWhkikwFfBM&t=2s)  
💻 **GitHub**: [DB-Vehicle-Database (private)](https://github.com/weiz-me/Vehicle-Database)

### Project Description
This project provides an interactive interface for managing a **vehicle sales and insurance database** using a Flask backend and PostgreSQL.

Core features include:
- 🔍 **Searchable Tables**: Query any table from the database with dynamic filtering
- 🚗 **Vehicle View**: Explore vehicles by manufacturer and price range
- 🏢 **Dealership & Insurance Lookup**: Filter dealerships and insurance by region and coverage
- 🛒 **Transaction Management**: Add new orders and track customer history
- 💬 **SQL Query Panel**: Run raw SQL queries for power users

Tech Stack:
- **Backend**: Flask + SQLAlchemy + PostgreSQL  
- **Frontend**: HTML templates, Jinja2, Bootstrap  
- **Deployment**: Hosted via Render.com

### Project Demo Video
""")

# Embed video
st.video("https://www.youtube.com/watch?v=gWhkikwFfBM&t=2s")

st.write("Project Site (3min bootup): https://vehicle-project.onrender.com/")

st.components.v1.iframe("https://vehicle-project.onrender.com/?embed=True", width=1500, height=1000, scrolling=True)

