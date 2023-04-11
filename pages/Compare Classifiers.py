import os
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from mtcnn import MTCNN

from helpers import get_marked_image, emotion_labels, get_prediction_of_own_CNN, images_folder_path, \
    setup_svm
from helpers import svm_get_predict, svm_model_exists, image_resizer

all_images = ["NA"]
group_images = ["NA"]
files = os.listdir(images_folder_path)
st.set_page_config(layout="wide")
svm_model_aws = setup_svm()

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = images_folder_path + "/" + file
        if file.startswith("group"):
            group_images.append(img_path)
        else:
            all_images.append(img_path)

group_tab, uploader_tab = st.tabs(["Group Images", "Upload your own image"])


def format_dictionary_probs(analysis):
    for k in analysis:
        analysis[k] = round(analysis[k], 5)
    return analysis


def get_dictonary_probs(probabilities):
    obj = {}
    for i, emotion_label in enumerate(emotion_labels):
        obj[emotion_label] = round(100 * probabilities[i], 5)
    return obj


@st.cache_data
def get_dictionary_probs_RMN(results):
    obj = {}
    for record in results[0]['proba_list']:
        for label in record:
            obj[label] = round(record[label] * 100, 5)

    return obj


with group_tab:
    group_image_path = st.selectbox(
        'Select an image:',
        group_images)

    if group_image_path != "NA":
        st.title("Here is the image you've selected")
        st.write(
            'Notice how the 2 face detection classifiers perform. On the right the neural network trained classifier is much better at finding faces')

        haar_column, dnn_column = st.columns(2)
        haar_column.subheader('HaarCascade face classifier')

        marked_image, nr_faces, faces_frames = get_marked_image(group_image_path)
        haar_column.metric(label="Faces found", value=nr_faces)
        haar_column.image(marked_image)

        dnn_column.subheader("neural network Face detector ")

        detector = MTCNN()

        img = cv2.imread(group_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = cv2.imread(group_image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        result = detector.detect_faces(img)
        faces_extracted = []
        for detection in result:
            x, y, width, height = detection['box']
            faces_extracted.append(original_img[y:y + height, x:x + width])
            # draw the rectangle box for face
            cv2.rectangle(img, (x, y), (width + x, height + y), (0, 212, 90), 3)

        dnn_column.metric(label="Faces found", value=len(faces_extracted))
        dnn_column.image(img)

        face_idx = 0
        btn = dnn_column.button("Calculate confidence scores", key=face_idx)
        for face in faces_extracted:
            obj, dominant_emotion = get_prediction_of_own_CNN(face)

            if face.shape[1] < 250 and face.shape[0] < 250:
                st.image(image_resizer(face, face.shape[1] * 2, face.shape[0] * 2))
            else:
                st.image(image_resizer(face, int(face.shape[1] / 2), int(face.shape[0] / 2)))

            if btn:

                expander2 = st.expander("See model classifications")

                col1, col2, col3 = expander2.columns(3)
                col1.subheader("Using own CNN model")
                col1.write('Dominant emotion ' + str(dominant_emotion))
                col1.write(obj)

                if face_idx <= len(faces_extracted):
                    col2.subheader('Using Deepface')

                    face_analysis = DeepFace.analyze(face, actions=['emotion'], detector_backend='opencv',
                                                     enforce_detection=False)
                    face_analysis[0]['emotion'] = format_dictionary_probs(face_analysis[0]['emotion'])

                    col2.write('Dominant emotion ' + str(face_analysis[0]['dominant_emotion']))
                    col2.write(face_analysis[0]['emotion'])
                    if svm_model_exists():
                        predicted_class, probabilities = svm_get_predict(svm_model_aws, face_img=face)

                        col3.subheader("Using SVM")
                        svm_dictionary = get_dictonary_probs(probabilities)
                        col3.write('Dominant emotion ' + str(predicted_class))
                        col3.write(svm_dictionary)

                face_idx += 1
    else:
        st.write("You can also select an image from above dropdown")

with uploader_tab:
    file_uploaded = st.file_uploader("Pick an image")

    if file_uploaded is not None:
        bytes_data = file_uploaded.getvalue()
        original_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if 'image' not in file_uploaded.type:
            st.warning("Please upload a JPG or PNG file.")
        else:

            detections = DeepFace.extract_faces(original_img, detector_backend="mtcnn", enforce_detection=False)
            faces_extracted = []
            for dictionary in detections:
                x = dictionary['facial_area']['x']
                y = dictionary['facial_area']['y']
                w = dictionary['facial_area']['w']
                h = dictionary['facial_area']['h']
                faces_extracted.append(original_img[y:y + h, x:x + w])

                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 215, 90), 2)
            confidence = detections[0]['confidence']
            if confidence > 0.5:
                st.image(original_img)
                st.write("Found the following faces:")
                for face in faces_extracted:
                    if face.shape[1] < 250 and face.shape[0] < 250:
                        st.image(image_resizer(face, face.shape[1] * 2, face.shape[0] * 2))
                    else:
                        st.image(image_resizer(face, int(face.shape[1] / 2), int(face.shape[0] / 2)))

                    obj, dominant_emotion = get_prediction_of_own_CNN(face)

                    # Deepface
                    analysis = DeepFace.analyze(face, actions=['emotion'], detector_backend='mtcnn',
                                                enforce_detection=False)

                    analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])

                    expander2 = st.expander("See models probabilities")

                    col1, col2, col3 = expander2.columns(3)

                    col1.subheader("Using own model")

                    col1.write(dominant_emotion)
                    col1.write(obj)
                    # SVM
                    if svm_model_exists():
                        col2.subheader("SVM")
                        predicted_class, probabilities = svm_get_predict(svm_model_aws, face_img=face)
                        svm_dictionary = get_dictonary_probs(probabilities)
                        col2.write("Dominant emotion: " + predicted_class)
                        col2.write(svm_dictionary)

                    col3.subheader("Deepface")
                    col3.write('Dominant emotion: ' + analysis[0]['dominant_emotion'])
                    col3.write(analysis[0]['emotion'])

            else:
                st.warning("No faces found! Try uploading another picture")
