import keras_preprocessing
import streamlit as st
from keras_preprocessing.image import load_img
import numpy as np
from PIL import Image

st.write( "CE301 - Capstone Project")

file = st.file_uploader("Pick an image")

mywidth = 48
if type(file) is not None:
    resized_img = Image.open(file)
    wpercent = (mywidth / float(resized_img.size[0]))
    hsize = int((float(resized_img.size[1]) * float(wpercent)))
    resized_img = resized_img.resize((mywidth, hsize))
    filename = 'resized.jpg'
    resized_img.save(filename)

from keras.models import load_model


# loading the model
loaded_model = load_model('best_model_optimised_cnn.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#  get the image, scale it to 48*48 and convert it to grayscale
resized_img = load_img('resized.jpg', target_size=(48, 48), color_mode="grayscale")

img_array = keras_preprocessing.image.img_to_array(resized_img)
img_array = np.expand_dims(img_array, axis=0)

image_input = np.vstack([img_array])

#  make predictions of the model
prediction = loaded_model.predict(image_input)
pred_argmaxed = prediction.argmax(axis=1)

labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

print(labels[np.argmax(prediction)])

st.title("Here is the image you've selected")
st.image(Image.open(file))

st.write("Prediction: ")

st.write(labels[np.argmax(prediction)])