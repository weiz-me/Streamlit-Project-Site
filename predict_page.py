import streamlit as st
import pickle
import numpy as np
import pandas as pd
import PIL

from matplotlib import pyplot as plt
from collections import defaultdict

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import Adam

'''
def load_c3():
    data = joblib.load("rf.joblib")
    return data

'''

def show_predict_page():

    st.write("#### Input your own images")
    uploaded_file = st. file_uploader("Upload your image (jpg, jpeg, png)...")

    img_model = InceptionV3(weights='imagenet') # This will download the weight files for you and might take a while.

    new_input = img_model.input
    new_output = img_model.layers[-2].output
    img_encoder = Model(new_input, new_output) # This is the final Keras image encoder model we will use.

    new_model = tf.keras.models.load_model('model/my_model')



    if uploaded_file:
        image = PIL.Image.open(uploaded_file)
        st.image(image)

        #resize image
        new_image = np.asarray(image.resize((299,299))) / 255.0
        encoded_image = img_encoder.predict(np.array([new_image]))

        #st.write(f"Greedy output: {image_decoder[]}")
