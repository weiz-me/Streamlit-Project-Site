import streamlit as st

st.set_page_config(page_title="AI-TicTacToe AI Games", page_icon="🎮")

st.write("""
# AI-TicTacToe AI Games 🎯🤖

### Project Links  
▶️ **Demo Website**: [https://ai-tictactoeai.onrender.com](https://ai-tictactoeai.onrender.com)  
▶️ **Demo Video**: [https://www.youtube.com/watch?v=zR3XVf887D0](https://www.youtube.com/watch?v=zR3XVf887D0)  

### Project Description
This application implements AI strategies for generalized m,n,k-games (Tic-Tac-Toe and its variants), using:

- 🔍 **Alpha-Beta Pruning Search**: Efficient minimax decision-making for adversarial games  
- 🌲 **Monte Carlo Tree Search (MCTS)**: Simulates random playouts and backpropagates outcomes to refine moves  
- 🧠 **Custom Game Loop**: Integrates human input and AI decisioning via Streamlit interface

Tech stack:
- `NumPy` and `Streamlit` for gameplay interface  
- Modular AI strategies implemented from scratch  
- Random, ABS, and MCTS-based strategies available per player

### Project Demo Video
""")

# Replace with your actual YouTube video URL
st.video("https://www.youtube.com/watch?v=zR3XVf887D0")

st.write("### Project Site: https://ai-tictactoeai.onrender.com")

# Embed the live demo site
st.components.v1.iframe("https://ai-tictactoeai.onrender.com", width=1500, height=800, scrolling=True)
