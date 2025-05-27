import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from collections import defaultdict
from PIL import Image

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from keras.layers import TFSMLayer

MAX_LEN = 40

# Load model using TFSMLayer (compatible with Keras 3+)
model2 = TFSMLayer("model/my_model2", call_endpoint="serving_default")

# Load word mappings
word_to_id = pickle.load(open("word_to_id.pkl", "rb"))
id_to_word = pickle.load(open("id_to_word.pkl", "rb"))

def image_decoder(enc_image):
    inputs = np.zeros((1, MAX_LEN))
    input_image = np.zeros((1, 2048))
    input_image[0] = enc_image

    index = 0
    inputs[0, index] = word_to_id["<START>"]
    index += 1

    while True:
        output = model2([input_image, inputs])[0].numpy()
        output_i = np.argmax(output[index - 1])
        output_word = id_to_word[output_i]
        inputs[0, index] = output_i
        index += 1

        if index == MAX_LEN or output_word == '<END>':
            break

    output_sentence_list = [id_to_word[o] for o in inputs[0] if o in id_to_word]
    output_sentence = " ".join(output_sentence_list)
    return output_sentence.replace("<PAD>", "")

def img_beam_decoder(n, image_enc):
    inputs = np.zeros((1, MAX_LEN))
    input_image = np.zeros((1, 2048))
    input_image[0] = image_enc

    inputs[0, 0] = word_to_id["<START>"]
    prob_seq = [(1, inputs)]

    for index in range(1, MAX_LEN):
        prob_seq_2 = []
        for p, s in prob_seq:
            if word_to_id["<END>"] in s[0]:
                prob_seq_2.append((p, s))
                continue
            output = model2([input_image, s])[0].numpy()
            output_topn_i = np.argsort(output[index - 1])[-n:]
            for i in output_topn_i:
                s1 = np.copy(s)
                s1[0, index] = i
                prob_seq_2.append((p * output[index - 1][i], s1))
        prob_seq = sorted(prob_seq_2, key=lambda x: x[0], reverse=True)[:n]

    output_list = []
    highest = -1000
    output_highest = ''
    for p, s in prob_seq:
        output_sentence_list = [id_to_word[o] for o in s[0] if o in id_to_word]
        output_sentence = " ".join(output_sentence_list).replace("<PAD>", "")
        if p > highest:
            highest = p
            output_highest = output_sentence
        output_list.append((p, output_sentence))
    return output_list, output_highest

def show_explore_page():
    st.title("Caption Generator")
    st.write("#### Explore Existing Dataset")

    FLICKR_PATH = "hw5data"

    def load_image_list(filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f]

    train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
    dev_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.devImages.txt'))
    test_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.testImages.txt'))

    st.write(f"""
    | Training data: {len(train_list)} |
    | Development data: {len(dev_list)} |
    | Testing data: {len(test_list)} |
    """)

    IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")
    img_num = st.selectbox("Pick a number between 0 and 7999", list(range(8000)))
    ok = st.button("View image and predict")

    def read_image_descriptions(filename):
        image_descriptions = defaultdict(list)
        with open(filename, 'r') as f:
            for line in f.read().strip().split('\n'):
                if "\t" in line:
                    title, caption = line.split("\t")
                    title = title.split("#")[0]
                    image_descriptions[title].append(caption.lower())
        return image_descriptions

    descriptions = read_image_descriptions(os.path.join(FLICKR_PATH, 'Flickr8k.token.txt'))
    OUTPUT_PATH = "hw5output"
    enc_train = np.load(os.path.join(OUTPUT_PATH, "encoded_images_train.npy"))
    enc_dev = np.load(os.path.join(OUTPUT_PATH, "encoded_images_dev.npy"))
    enc_test = np.load(os.path.join(OUTPUT_PATH, "encoded_images_test.npy"))

    if ok:
        if img_num < 6000:
            select = train_list[img_num]
            encoded = enc_train[img_num]
        elif img_num < 7000:
            select = dev_list[img_num - 6000]
            encoded = enc_dev[img_num - 6000]
        else:
            select = test_list[img_num - 7000]
            encoded = enc_test[img_num - 7000]

        image = Image.open(os.path.join(IMG_PATH, select))
        st.image(image)

        output1 = image_decoder(encoded).replace("<START>", "").replace("<END>", "")
        output2 = img_beam_decoder(3, encoded)[0][0][1].replace("<START>", "").replace("<END>", "")

        df = pd.DataFrame({
            "Model type": ["Dataset Value", "Greedy Output", "Beam Search n=5"],
            "Output": [descriptions[select][0], output1, output2]
        })

        st.dataframe(df)
        st.write("The model has 40% accuracy")

def show_predict_page():
    st.write("#### Input your own images")
    uploaded_file = st.file_uploader("Upload your image (jpg, jpeg, png)...")

    img_model = InceptionV3(weights='imagenet')
    new_input = img_model.input
    new_output = img_model.layers[-2].output
    img_encoder = Model(new_input, new_output)

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image)

        new_image = np.asarray(image.resize((299, 299))) / 255.0
        encoded_image = img_encoder.predict(np.array([new_image]))

        output11 = image_decoder(encoded_image).replace("<START>", "").replace("<END>", "")
        output22 = img_beam_decoder(3, encoded_image)[0][0][1].replace("<START>", "").replace("<END>", "")

        df = pd.DataFrame({
            "Model type": ["Greedy Output", "Beam Search n=5"],
            "Output": [output11, output22]
        })

        st.dataframe(df)
        st.write("The model has 40% accuracy")
