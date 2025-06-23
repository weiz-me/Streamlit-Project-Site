import streamlit as st

st.set_page_config(page_title="Caption Generator Project", page_icon="🖼️")

st.write("""
# Image Caption Generator 🖼️📝

### Project Links  
▶️ **Demo Video**: [https://www.youtube.com/watch?v=Qd1XGognnS0](https://www.youtube.com/watch?v=Qd1XGognnS0)  

### Project Description
This application demonstrates a deep learning-based image caption generator using:
- **CNN + LSTM architecture** for image-to-text generation
- **Greedy decoding and Beam Search** for caption generation strategies
- **Transfer learning** with InceptionV3 for image feature extraction

The model was trained on the Flickr8k dataset and supports:
- Exploring training, dev, and test image samples with generated captions
- Uploading your own image to generate captions

It uses:
- `TensorFlow`, `Keras`, `NumPy`, `PIL`, and `Streamlit`  
- Pretrained `InceptionV3` model for image encoding  
- Custom-trained LSTM model for caption decoding

### Project Demo Video
""")

# Replace with your actual YouTube video ID or leave it as-is for now
st.video("https://www.youtube.com/watch?v=Qd1XGognnS0")
