import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import os


import os
from collections import defaultdict
import numpy as np
import PIL
from matplotlib import pyplot as plt

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras


# model2 = tf.keras.models.load_model('model/my_model2')
model2 =keras.layers.TFSMLayer('model/my_model2', call_endpoint='serving_default')
MAX_LEN = 40
word_to_id = pickle.load(open("word_to_id.pkl","rb"))
id_to_word = pickle.load(open("id_to_word.pkl","rb"))

def image_decoder(enc_image): 
    inputs=np.zeros((1,MAX_LEN))
    input_image = np.zeros((1,2048))
    input_image[0]=enc_image
    
    index = 0
    inputs[0,index] = word_to_id["<START>"]
    index +=1
    
    while True:
        output = model2.predict([input_image,inputs],verbose= False)[0]
        #print(output)
        output_i = np.argmax(output)
        output_word = id_to_word[output_i]
        inputs[0,index] = output_i
        index+=1
        
        if index == 40 or output_word =='<END>':
            break
    
    output_id = inputs
    output_sentence_list = [id_to_word[o] for output in output_id for o in output ]
    output_sentence=" ".join(output_sentence_list)
    output_sentence=output_sentence.replace("<PAD>","")
    
    return output_sentence

def img_beam_decoder(n, image_enc):
    inputs=np.zeros((1,MAX_LEN))
    input_image = np.zeros((1,2048))
    input_image[0]=image_enc
    
    index = 0
    inputs[0,index] = word_to_id["<START>"]
    
    prob_seq = []
    prob_seq.append((1,inputs))
    
    for index in range(1,MAX_LEN):
        prob_seq_2 = []
        
        for p,s in prob_seq:
            if word_to_id["<END>"] in s[0]:
                prob_seq_2.append((p,s))
                continue
            
            output = model2.predict([input_image,s],verbose= False)[0]
            output_topn_i = np.argsort(output)[-n:]
            #print(output_topn_i)
            for i in output_topn_i:
                #print(id_to_word[i])
                s1=np.copy(s)
                s1[0,index]=i
                #print(s1)
                #print(output[i])
                prob_seq_2.append((p*output[i],s1))
            
        
        prob_seq = sorted(prob_seq_2, key=lambda x: x[0], reverse=True)[:n]
        #print(f"prob_seq_2",prob_seq_2)
        #print(f"prob_seq",prob_seq)
        
    output_list=[]
    highest = -1000
    output_highest=''
    for p,s in prob_seq:
        output_id = s[0]
        output_sentence_list = [id_to_word[o] for o in output_id]
        output_sentence=" ".join(output_sentence_list)
        output_sentence=output_sentence.replace("<PAD>","")
        
        if p>highest:
            highest = p
            output_highest = output_sentence
        
        output_list.append((p,output_sentence))
    return output_list,output_sentence



def show_explore_page():
    st.title("Caption Generator")

    st.write(
        """
    #### Explore Existing Dataset """)

    #loading model

    FLICKR_PATH="hw5data"

    def load_image_list(filename):
        with open(filename,'r') as image_list_f: 
            return [line.strip() for line in image_list_f]    
    
    train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
    dev_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.devImages.txt'))
    test_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.testImages.txt'))

    st.write(f"""|
training data : {len(train_list)} |
developing data : {len(dev_list)}|
testing data : {len(test_list)}|
             """)

    import tensorflow as tf
    IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")

    nums_list = [i for i in range(8000)]
    img_num = st.selectbox("Pick a number between 0 and 7999", nums_list)
    ok = st.button("View image and predict")

    def read_image_descriptions(filename):    
        image_descriptions = defaultdict(list)    
        with open(filename, 'r') as f:
            file_contents = f.read()

        line_contents = file_contents.split('\n')

        #print(line_contents[0])
        for line_content in line_contents:
            line_token=line_content.split("\t")
            title = line_token[0].split("#")[0]
            #print(title)
            tokens=line_token[-1].lower()
            #tokens=['<START>']+tokens + ['<END>']
            
            image_descriptions[title].append(tokens)
        return image_descriptions
    
    descriptions = read_image_descriptions(f"{FLICKR_PATH}/Flickr8k.token.txt")
    OUTPUT_PATH = "hw5output" 
    enc_train = np.load(os.path.join(OUTPUT_PATH,"encoded_images_train.npy"))
    enc_dev = np.load(os.path.join(OUTPUT_PATH,"encoded_images_dev.npy"))
    enc_test = np.load(os.path.join(OUTPUT_PATH,"encoded_images_test.npy"))





    if ok:
        if img_num <6000:
            select = train_list[img_num]
            select2 = enc_train[img_num]
        
        elif img_num <7000:
            select = dev_list[img_num-6000]
            select2 = enc_dev[img_num-6000]

        else:
            select = test_list[img_num-7000]
            select2 = enc_test[img_num-7000]

        image = PIL.Image.open(os.path.join(IMG_PATH, select))
        st.image(image)
        
        #loading model

        output1 = image_decoder(select2)
        output1 = output1.replace("<START>","").replace("<END>","")
        output2 = img_beam_decoder(3, select2)[0][0][-1]
        output2 = output2.replace("<START>","").replace("<END>","")

        df2 = {"Model type":["Dataset Value","Greedy Output","Beam Search n=5"],
               "Output": [descriptions[select][0],output1,output2]}

        st.dataframe(df2)
        st.write("The model has 40\% accuracy")





# def show_predict_page():

#     st.write("#### Input your own images")
#     uploaded_file = st. file_uploader("Upload your image (jpg, jpeg, png)...")

#     img_model = InceptionV3(weights='imagenet') # This will download the weight files for you and might take a while.

#     new_input = img_model.input
#     new_output = img_model.layers[-2].output
#     img_encoder = Model(new_input, new_output) # This is the final Keras image encoder model we will use.


#     if uploaded_file:
#         image = PIL.Image.open(uploaded_file)
#         st.image(image)

#         #resize image
#         new_image = np.asarray(image.resize((299,299))) / 255.0
#         encoded_image = img_encoder.predict(np.array([new_image]))


#         output11 = image_decoder(encoded_image)
#         output11 = output11.replace("<START>","").replace("<END>","")
#         output22 = img_beam_decoder(3, encoded_image)[0][0][-1]
#         output22 = output22.replace("<START>","").replace("<END>","")

#         df22 = {"Model type":["Greedy Output","Beam Search n=5"],
#                "Output": [output11,output22]}

#         st.dataframe(df22)
#         st.write("The model has 40\% accuracy")


