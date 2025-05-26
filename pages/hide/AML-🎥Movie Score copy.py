import streamlit as st
#from predict_page1 import show_predict_page
#from explore_page1 import show_explore_page


#page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
st.set_page_config(
    page_icon="🧑‍💻"
)


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

def load_model1():
    with open('model1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_model2():
    with open('model2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_model3():
    with open('model3.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_c1():
    with open('c1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_c2():
    with open('c2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_c3():
    data = joblib.load("rf.joblib")
    return data





def load_y():
    with open('y.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_y2():
    with open('y2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_csr_x_1():
    with open('csrx1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_csr_x_2():
    with open('csrx2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data




model1 = load_model1()
model2 = load_model2()
model3 = load_model3()
c1 = load_c1()
c2 = load_c2()
c3 = load_c3()

from tensorflow.keras.models import load_model

nn = load_model("nn.h5")

y = load_y()
y2 = load_y2()
csrx1 = load_csr_x_1()
csrx2 = load_csr_x_2()

def show_predict_page():

    st.title("Movie Score Prediction")

    st.write("""### Select Movie""")
    movie1_df = pd.read_csv('https://weizhang45.com/projects/data/movie1.csv')

    movies = st.selectbox("movie", movie1_df["title"].tolist())
    ok = st.button("Predict Score")

    if ok:
        df=movie1_df[movie1_df["title"]==movies]
        n = df.index[0]
        st.dataframe(df)
        st.write("### Movie rating predictions")

        df1 = {"Model type":["Dataset Value","Linear Regression","Linear Reg w/ Lasso","Linear Reg w/ Ridge"],
               "Rating": [y[n],model1.predict(csrx1[n,:])[0],model2.predict(csrx1[n,:])[0],model3.predict(csrx1[n,:])[0]]}
               
        st.dataframe(df1)



        st.write("### Movie Popularity predictions")


        df2 = {"Model type":["Dataset Value","Logistic Regression","Decision Tree","Random Forest","Neural Network"],
               "Popular ?": [y2[n],c1.predict(csrx1[n,:])[0],c2.predict(csrx2[n,:])[0],c3.predict(csrx2[n,:])[0],round(nn.predict(csrx2[n,:])[0][0])]}
               
        st.dataframe(df2)

show_predict_page()

st.write("\n\n\n### view report detail")
a=st.button("View data preparation, analysis, and model result")

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

def show_explore_page():
    st.title("Explore Movie Dataset")

    st.write(
        """
    ## P1: Initial Data Exploration""")

    st.write("""
    #### The movie dataset contains following 19 features:
filmtv_id, title, year, genre, duration, country,directors, actors, avg_vote, critics_vote, public_vote,total_votes, description, notes, humor, rhythm, effort, tension, erotism.

    """
    )

    
    movie1_df = pd.read_csv('https://weizhang45.com/projects/data/movie1.csv')
    
    st.dataframe(movie1_df)

    st.write("""
    #### Missing Data
Most of the features have no missing values except 8 variables which have missing values. 
These variables are genre, country, directors, actors, critics_vote, public_vote, description, and notes.
To work with the missing values, we will drop notes, description, public votes, and critics_votes.
We will work with genre, country, directors, and actors. These are important features.
s
    """
    )

    missing_values_per_column = movie1_df.isna().sum() 
    #print(missing_values_per_column)
    movie1_df.fillna('missing',inplace=True)    
        
    plt.barh(y=missing_values_per_column.index, width = missing_values_per_column)
    plt.title('Histogram of missing values per column')
    #plt.show()

    st.pyplot(plt)

    movie1_df['directors'] = movie1_df['directors'].str.replace(", ",",").str.replace(" ,",",").str.strip(" ").str.split(',')
    movie1_df['actors'] = movie1_df['actors'].str.replace(", ",",").str.replace(" ,",",").str.strip(" ").str.split(',')
    movie1_df['country'] = movie1_df['country'].str.replace(", ",",").str.replace(" ,",",").str.strip(" ").str.split(',')
    movie1_df = movie1_df.drop(['critics_vote', 'public_vote', 'total_votes'], axis = 1)
    median_score = movie1_df['avg_vote'].median()
    movie1_df['popular'] = movie1_df['avg_vote'].apply(lambda x: 1 if x > median_score else 0)


    plt.figure(figsize = (10, 10))
    count = movie1_df['popular'].value_counts()
    plt.bar(count.index, count.values)
    plt.title("The Distribution of Target Variable")
    plt.xlabel("popularity")
    plt.ylabel("count")
    
    st.write("""
    #### Dataset Balance  
We use mean rating of 5.9 as cut-off score for popularity.
    """
    )

    st.pyplot(plt)

    cate_columns = ["rhythm", "effort", "tension", "erotism"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    j=-1
    for i, cate in enumerate(cate_columns):
        if i % 2 == 0:
            j +=1
        
        grouped = movie1_df.groupby([cate, 'popular']).size().unstack()
        grouped.plot(kind='bar', stacked=False, ax=axes[i%2,j])
        axes[i%2,j].set_title(f"Class Distribution by {cate} and Target Categories")
        axes[i%2,j].set_xlabel(cate)
        axes[i%2,j].set_ylabel('Count')
        axes[i%2,j].set_xticklabels(axes[i%2,j].get_xticklabels(), rotation=0)

        
    plt.tight_layout()
    #plt.show()
    st.write("""
##### Discrete Variable target Distribution
    """
    )
    st.pyplot(fig)

    st.write("""
Note: director, actor, and country contains too many entries and were not able to plot.
    """
    )

    st.write("""
##### Continuous Variable Distribution
The continuous features are year, duration, 
avg_vote.
             
○ Most of the movies is released 
recently after 2000\n
○ Average duration of movie is around 
90 minutes.\n
○ The median value of avg_vote is 5.9 / 
10.0

    """
    )
    fig, axes = plt.subplots(nrows=2, ncols =2)

    i=0
    j=0
    for category in ["year", "duration","avg_vote"]:

        if category == "duration":
            movie1_df.hist(category,ax=axes[j,i],bins=100)

            axes[j,i].set_xlim([0,170])

            i+=1
            if i == 2:
                j+=1
                i=0
            continue



        movie1_df.hist(category,ax=axes[j,i])
        
        i+=1
        if i == 2:
            j+=1
            i=0

    st.pyplot(fig)


    plot_df=movie1_df[['year', 'genre', 'duration','humor', 'rhythm', 'effort','tension', 'erotism', 'popular','avg_vote']]
    plot_list=['year', 'genre', 'duration','humor', 'rhythm', 'effort','tension', 'erotism']

    fig, axes = plt.subplots(nrows=2, ncols =2)
    i=0
    j=0
    for col in plot_list[:4]:
            
        plot_df.plot(x=col,y="avg_vote",kind = "scatter",ax=axes[i,j])
        i+=1
        if i ==2:
            j+=1
            i=0

    fig.tight_layout(pad=1.0)
    st.pyplot(fig)








    st.write("""
    # P2: Cleaning and Sampling
        """
        )
    st.write("""
    #### Cleaning\n
        ○ Dropping high correlation features: critics_vote and public vote
        ○ Dropping unnecessary features: filmtv_id, total_votes, descriptions, and notes.

        """
        )
    st.image("./img/d1.jpg")


    st.write("""
        #### Cleaning\n
            ○ One-hot encoding countries, directors, actors (eliminate values with less than 7).
            ○ Category-encoding genre.
             

            """
            )

        
    def load_dataclean():
        with open('data_clean.pkl', 'rb') as file:
            data = pickle.load(file)
        return data

    dataclean = load_dataclean()

    selected_columns = dataclean.iloc[:, list(range(0, 10))]
    st.dataframe(selected_columns)
    
    st.write("""
Countries, actors, director (one-hot encoding) were not show, due limited space. Screenshot below        """
        )
    st.image("./img/d2.jpg")

    
    
    st.write("""
    # P3: Model
        """
        )
    
    
    st.write("""
        ○ 60% traning / 20% validation / 20% testing data
  

        """
        )
    
    st.write("""
    #### Regression models (predict movie rating)\n
        ○ Linear regression model
        ○ Linear regression with Lasso
        ○ Linear regression with Ridge
        ○ y_var = 2.0


        """
        )
    with open('s1.pkl', 'rb') as f:
        x = pickle.load(f)
    st.write("\n")

    st.dataframe(x)


    st.write("""
    #### Classification models (predict movie popularity)\n
        ○ Logistic regression 
        ○ Decision tree 
        ○ Random forest 
        ○ Nerual Network 

        """
        )
    
    
    with open('s2.pkl', 'rb') as f:
        x = pickle.load(f)
    st.write("\n")

    st.dataframe(x)



    st.write("""
        #### Top 5 Features for each model\n
        """
        )

    
    with open('s3.pkl', 'rb') as f:
        x = pickle.load(f)
    st.write("\n")

    st.dataframe(x)
    
if a:
    show_explore_page()

