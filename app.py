import streamlit as st
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
from keras.models import Sequential
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import random
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage import io

data = pd.read_csv('german-traffic-signs/signnames.csv')

def grayscale(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    return image

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

new_model = tf.keras.models.load_model('/content/signs.h5')



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.title("Sign Detection")
url = st.text_input("**Enter a traffic sign here** ", "Start typing here")
if st.button("**Get Result**"):
    r = requests.get(url, stream=True)
    image = Image.open(r.raw)

    img = np.asarray(image)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    imge = img.reshape(1, 32, 32, 1)
    new_model.predict(imge)
    predictions=(np.argmax(new_model.predict(imge)))   
    pred = np.round(predictions).astype(int)
    
  
    fig = make_subplots(rows=1, cols=2 ,shared_yaxes=False)
    fig.add_trace(go.Image(z=image), 1, 1)
    fig.add_trace(go.Heatmap(z=img, colorscale='gray'),1,2).update_yaxes(autorange='reversed',constrain='domain').update_xaxes(constrain='domain')
  
    fig

   
    for num, name in data.iteritems():
      name = name.values
      st.write(f"**predicted sign : {str(name[pred])}**")
