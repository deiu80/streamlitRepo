import pathlib
import cv2
import streamlit as st
from PIL import Image
import os
from helpers import face_detect_NN, get_marked_image, emotion_labels
from helpers import images_folder_path



st.title("Image uploader")

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

# list files in img directory
files = os.listdir(images_folder_path)
all_images = ["NA"]
group_images = ["NA"]


for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = images_folder_path + "/" + file
        if file.startswith("group"):
            group_images.append(img_path)
        else:
            all_images.append(img_path)

file_uploaded = st.file_uploader("Pick an image")

if file_uploaded is not None:
    st.title("Here is the original image you've uploaded")
    original_img = Image.open(file_uploaded)
    uploaded_file_path = images_folder_path + "/original" + file_uploaded.name
    original_img.save(uploaded_file_path)
    file_uploaded.close()

    st.image(original_img)
    haar_column, dnn_column = st.columns(2)

    haar_column.subheader('HaarCascade classifier example')
    marked_image, nr_faces, faces_frames = get_marked_image(uploaded_file_path)
    haar_column.metric(label="Faces found", value=nr_faces)

    haar_column.image(marked_image)

    dnn_column.subheader("Classifier pre-trained on Res10 ")
    annotated_img, nr_faces2, faces_extracted = face_detect_NN(uploaded_file_path, 0.7)
    dnn_column.metric(label="Faces found", value=nr_faces2)
    dnn_column.image(annotated_img)