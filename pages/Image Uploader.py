import pathlib
import cv2
import streamlit as st
from keras_preprocessing.image import load_img
from PIL import Image
import os
from helpers import get_model_prediction
from helpers import get_img_face_frame
from helpers import images_folder_path
st.set_page_config(
    page_title="CE301 - Image uploader"
)
st.title("Image uploader")

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

# list files in img directory
files = os.listdir(images_folder_path)
all_images = ["NA"]
labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = images_folder_path + "/" + file
        all_images.append(img_path)

file_uploaded = st.file_uploader("Pick an image")

if file_uploaded is not None:
    original_img = Image.open(file_uploaded)
    original_img.save("./original.jpg")
    file_uploaded.close()
else:
    st.write("No image selected")

selected_image_path = st.sidebar.selectbox("Images Paths", all_images)

if selected_image_path != "NA":
    # original image
    st.title("Here is the image you've selected")
    st.image(selected_image_path)

    # extracting the face from image
    face_frame = get_img_face_frame(selected_image_path)
    st.write("Extracted face is: ")
    st.image(face_frame)

    # PIL Image instance.
    resized_face_img = load_img(images_folder_path + '/extracted_face.jpg', target_size=(48, 48), color_mode="grayscale")
    emotion_predicted = get_model_prediction(resized_face_img)
    st.write("Model's prediction: " + emotion_predicted)
else:
    if file_uploaded is not None:
        # get original image
        #  get face from the image, scale it to 48*48 and convert it to grayscale
        st.title("Here is the image you've uploaded")
        img = cv2.imread("./original.jpg")
        st.image(img)
        # extracting the face from image
        face_frame = get_img_face_frame("./original.jpg")
        st.write("Extracted face is: ")
        st.image(face_frame)

        resized_face_img = load_img(images_folder_path + '/extracted_face.jpg', target_size=(48, 48), color_mode="grayscale")
        emotion_predicted = get_model_prediction(resized_face_img)
        st.write("Model's prediction: "+ emotion_predicted)


agree = st.checkbox('Show the last extracted face')

if agree:
    st.write("Most recent extracted face is:")
    st.image(load_img(images_folder_path + '/extracted_face.jpg'))