import pathlib
import cv2
import streamlit as st
from keras_preprocessing.image import load_img
from PIL import Image
import os
from helpers import get_model_prediction, get_all_faces, image_resizer, face_detect_NN, get_marked_image, _max_width_, \
    labels
from helpers import images_folder_path

_max_width_()

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

group_image_box = st.checkbox("Use group images")

if group_image_box:
    group_image_path = st.sidebar.selectbox("Group Images", group_images)
    if group_image_path != "NA":
        st.title("Here is the image you've selected")
        rgb_img = cv2.cvtColor(cv2.imread(group_image_path), cv2.COLOR_BGR2RGB)
        st.image(rgb_img)
        haar_column, dnn_column = st.columns(2)
        haar_column.subheader('HaarCascade classifier example')

        marked_image, nr_faces, faces_frames = get_marked_image(group_image_path)
        haar_column.metric(label="Faces found", value=nr_faces)
        haar_column.image(marked_image)

        for el in faces_frames:
            gray_el = cv2.cvtColor(el, cv2.COLOR_RGB2GRAY)
            pil_image = Image.fromarray(gray_el)
            pil_image = pil_image.resize((48, 48))
            emotion_predicted, scores = get_model_prediction(pil_image)
            haar_column.write("Model's prediction: " + emotion_predicted)
            haar_column.image(el)

        dnn_column.subheader("Classifier pre-trained on Res10 ")
        annotated_img, nr_faces2, faces_extracted = face_detect_NN(group_image_path, 0.7)
        dnn_column.metric(label="Faces found", value=nr_faces2)
        dnn_column.image(annotated_img)
        for el in faces_extracted:
            gray_el = cv2.cvtColor(el, cv2.COLOR_RGB2GRAY)
            pil_image = Image.fromarray(gray_el)
            pil_image = pil_image.resize((48, 48))
            emotion_predicted, scores = get_model_prediction(pil_image)
            dnn_column.write("Model's prediction: " + emotion_predicted)
            dnn_column.image(el)
            expander2 = dnn_column.expander("See models probabilities")
            for lab, score in zip(labels, scores):
                expander2.metric(value=score, label=lab)
    else:
        st.write("You can also select an image from left bar")

single_images_box = st.checkbox('Use singular images')

if single_images_box:
    selected_image_path = st.sidebar.selectbox("Singular subjects", all_images)
    if selected_image_path != "NA":
        st.title("Here is the original image you've selected")
        rgb_img = cv2.cvtColor(cv2.imread(selected_image_path), cv2.COLOR_BGR2RGB)
        st.image(rgb_img)
        haar_column, dnn_column = st.columns(2)
        haar_column.subheader('HaarCascade classifier example')
        marked_image, nr_faces, faces_frames = get_marked_image(selected_image_path)
        haar_column.metric(label="Faces found", value=nr_faces)
        haar_column.image(marked_image)

        for el in faces_frames:
            gray_el = cv2.cvtColor(el, cv2.COLOR_RGB2GRAY)
            pil_image = Image.fromarray(gray_el)
            pil_image = pil_image.resize((48, 48))
            emotion_predicted, scores = get_model_prediction(pil_image)
            haar_column.write("Model's prediction: " + emotion_predicted)
            haar_column.image(el)
            expander = haar_column.expander("See explanation")
            expander.write("classes probabilities")
            expander.write(scores)

        dnn_column.subheader("Classifier pre-trained on Res10 ")
        annotated_img, nr_faces2, faces_extracted = face_detect_NN(selected_image_path, 0.7)
        dnn_column.metric(label="Faces found", value=nr_faces2)
        dnn_column.image(annotated_img)

        for el in faces_extracted:
            gray_el = cv2.cvtColor(el, cv2.COLOR_RGB2GRAY)
            pil_image = Image.fromarray(gray_el)
            pil_image = pil_image.resize((48, 48))
            emotion_predicted, scores = get_model_prediction(pil_image)
            dnn_column.write("Model's prediction: " + emotion_predicted)
            dnn_column.image(el)


    else:
        st.write("You can also select an image from left bar")

if file_uploaded is not None:
    st.title("Here is the original image you've uploaded")
    original_img = Image.open(file_uploaded)
    uploaded_file_path = images_folder_path + "/original" + file_uploaded.name
    original_img.save(uploaded_file_path)
    file_uploaded.close()

    st.image(original_img)
    haar_column, dnn_column = st.columns(2)

    haar_column.subheader('HaarCascade classifier example')
    print(uploaded_file_path)
    marked_image, nr_faces, faces_frames = get_marked_image(uploaded_file_path)
    haar_column.metric(label="Faces found", value=nr_faces)

    haar_column.image(marked_image)

    dnn_column.subheader("Classifier pre-trained on Res10 ")
    annotated_img, nr_faces2, faces_extracted = face_detect_NN(uploaded_file_path, 0.7)
    dnn_column.metric(label="Faces found", value=nr_faces2)
    dnn_column.image(annotated_img)
