import pathlib

import cv2
import keras_preprocessing
import matplotlib.pyplot as plt
import streamlit as st
from keras_preprocessing.image import load_img
import numpy as np
from PIL import Image
import os
from keras.models import load_model

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)
clf = cv2.CascadeClassifier(str(cascade_path))

# specify the img directory path
path = "./Images"

# list files in img directory
files = os.listdir(path)

all_images = ["NA"]
for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = path + "/" + file
        all_images.append(img_path)


file = st.file_uploader("Pick an image")

mywidth = 48
if file is not None:
    original_img = Image.open(file)
    wpercent = (mywidth / float(original_img.size[0]))
    hsize = int((float(original_img.size[1]) * float(wpercent)))
    resized_img = original_img.resize((mywidth, hsize))

    filename = './resized.jpg'
    resized_img.save(filename)
    original_img.save("./original.jpg")
    file.close()
else:
    st.write("No image selected")

selected_image = st.sidebar.selectbox("Image Name", all_images)

loaded_model = load_model('./best_model_optimised_cnn.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if selected_image != "NA":
    resized_img = load_img(selected_image, target_size=(48, 48), color_mode="grayscale")
    img_array = keras_preprocessing.image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    image_input = np.vstack([img_array])
    #  make predictions of the model
    prediction = loaded_model.predict(image_input)

    pred_argmaxed = prediction.argmax(axis=-1)

    labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

    st.title("Here is the image you've selected")
    st.image(selected_image)

    st.write("Prediction: ")

    st.write(labels[np.argmax(prediction)])
else:
    if file is not None:
        # loading the model
        #  get the image, scale it to 48*48 and convert it to grayscale

        resized_img = load_img('./resized.jpg', target_size=(48, 48), color_mode="grayscale")

        img_array = keras_preprocessing.image.img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)

        image_input = np.vstack([img_array])
        #  make predictions of the model
        prediction = loaded_model.predict(image_input)
        print("Prediction:", prediction[0])
        print(prediction.shape)
        pred_argmaxed = prediction.argmax(axis=-1)
        print(pred_argmaxed)

        labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
        print(labels[np.argmax(prediction)])

        st.title("Here is the image you've selected")
        st.image(load_img("./original.jpg"))

        st.write("Prediction: ")

        st.write(labels[np.argmax(prediction)])
