import pathlib
import cv2
import streamlit as st
import keras_preprocessing
from keras_preprocessing.image import load_img
import numpy as np

from keras.models import load_model

labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

images_folder_path = "./Images"


def get_model_prediction(image):
    img_array = keras_preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    image_input = np.vstack([img_array])
    loaded_model = load_model('./best_model_optimised_cnn.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #  make predictions of the model
    prediction = loaded_model.predict(image_input)

    return labels[np.argmax(prediction)]


def get_img_face_frame(img_path):
    nr_of_faces = 0
    cv2_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray_img,
                                 scaleFactor=1.2,
                                 minNeighbors=5,
                                 minSize=(30, 30),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, width, height) in faces:
        cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_frame = gray_img[y:height + y, x:x + width]
        nr_of_faces += 1
    if faces != ():
        cv2.imwrite(images_folder_path + "/extracted_face.jpg", face_frame)
    st.write("Nr of faces found: " + str(nr_of_faces))
    return cv2.imread(images_folder_path + "/extracted_face.jpg")
