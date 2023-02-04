import pathlib
from os.path import exists
import streamlit as st
import cv2
from keras_preprocessing.image import load_img
from helpers import get_model_prediction, image_enhancer, get_classifier
import numpy as np

st.set_page_config(
    page_title="CE301 - Webcam Live Feed"
)

classifier = get_classifier()
st.title("Webcam Live Feed")

img_file_buffer = st.camera_input("Take a frontal picture")
path = "./Images"


if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img,
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(30, 30),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, width, height) in faces:
        cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_frame = gray_img[y:height + y, x:x + width]
        enhanced_image = image_enhancer(face_frame)
        # cv2.imwrite(path + "/enhanced_face.jpg", enhanced_image)
        cv2.imwrite(path + "/extracted_face.jpg", face_frame)

    if exists(path + "/extracted_face.jpg"):
        st.write("Extracted face is:")
        st.image(cv2.imread(path + "/extracted_face.jpg"))

        resized_img = load_img(path + "/extracted_face.jpg", target_size=(48, 48), color_mode="grayscale")

        enhanced_image = load_img(path + "/enhanced_face.jpg", target_size=(48, 48), color_mode="grayscale")
        emotion_predicted = get_model_prediction(enhanced_image)
        st.write("Model's prediction: " + emotion_predicted)
