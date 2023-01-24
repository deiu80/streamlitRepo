import pathlib
from os.path import exists

import numpy as np
import streamlit as st
import cv2
from keras_preprocessing.image import load_img
from helpers import get_model_prediction

labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

st.title("Webcam Live Feed")

img_file_buffer = st.camera_input("Take a frontal picture")
path = "./Images"

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    # st.write(type(cv2_img))

    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray_img,
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(30, 30),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, width, height) in faces:
        cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_frame = gray_img[y:height + y, x:x + width]
    cv2.imwrite(path + "/extracted_face.jpg", face_frame)


    if exists(path + "/extracted_face.jpg"):
        st.write("Extracted face is:")
        st.image(cv2.imread(path + "/extracted_face.jpg"))
        resized_img = load_img(path + "/extracted_face.jpg", target_size=(48, 48), color_mode="grayscale")
        emotion_predicted = get_model_prediction(resized_img)
        st.write("Model's prediction: " + emotion_predicted)

