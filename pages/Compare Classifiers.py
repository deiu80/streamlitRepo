import os

import cv2
import streamlit as st
from PIL import Image
from deepface import DeepFace
from rmn import RMN

from helpers import face_detect_NN, get_marked_image, emotion_labels, faces_folder_path, get_prediction_deepface_way, \
    svm_get_predict, setup_svm, prepare_image_for_svm, svm_model_exists,single_face_img_path, example_face_img_path
from helpers import images_folder_path

all_images = ["NA"]
group_images = ["NA"]
files = os.listdir(images_folder_path)

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg')):
        img_path = images_folder_path + "/" + file
        if file.startswith("group"):
            group_images.append(img_path)
        else:
            all_images.append(img_path)


svm_model_aws = setup_svm()

tab1, tab2, tab3 = st.tabs(["Example", "Group Images", "Single Images"])

labels = {
    "top_left": "CNN model: ",
    "top_right": "SVM model: ",
    "bottom_left": "Deepface: ",
    "bottom_right": "ResMaskNet: "
}


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


@st.cache_resource
def load_RMN():
    rmn = RMN()
    return rmn


def annotate_example_image(img, labels):
    label_positions = {
        "top_left": (10, 20),
        "top_right": (img.shape[1] - 5 - len(labels["top_right"]) * 11, 20),
        "bottom_left": (10, img.shape[0] - 10),
        "bottom_right": (img.shape[1] - 10 - len(labels["bottom_right"]) * 11, img.shape[0] - 10)
    }
    # Define the font and font scale to use for the labels
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6

    # Loop through each label and draw it on the image
    for label, text in labels.items():
        cv2.putText(img, text, label_positions[label], font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img


with tab1:
    st.title("Compare the classifiers ")
    st.write("We do have 4 models available: own CNN model,own SVM, DeepFace and ResMasknet ")

    returned_img, nr, faces_extracted = face_detect_NN(single_face_img_path, 0.7)

    # OWN CNN
    obj, dominant_emotion = get_prediction_deepface_way(returned_img, example_face_img_path)
    labels['top_left'] += dominant_emotion + " " + str(round(obj[dominant_emotion], 2))
    # Deepface
    analysis = DeepFace.analyze(single_face_img_path, actions=['emotion'], detector_backend='mtcnn')
    analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])
    dominant_emotion_deepface = analysis[0]['dominant_emotion']
    labels["bottom_left"] += analysis[0]['dominant_emotion'] + " " + str(
        round(analysis[0]['emotion'][dominant_emotion_deepface], 2))

    # SVM prediction
    predicted_class, probabilities = svm_get_predict(example_face_img_path, svm_model_aws)
    svm_dictionary = get_dictonary_probs(probabilities)
    labels['top_right'] += predicted_class + ' ' + str(round(svm_dictionary[predicted_class], 2))

    # RMN prediction

    rmn = load_RMN()
    im = cv2.imread(single_face_img_path)
    results = rmn.detect_emotion_for_single_frame(im)
    rmn_dict_proba = get_dictionary_probs_RMN(results)
    labels['bottom_right'] += results[0]['emo_label'] + ' ' + str(round(rmn_dict_proba[results[0]['emo_label']], 2))

    annoted_image = annotate_example_image(returned_img, labels)
    st.image(annoted_image)
    st.write("You can notice the predicted enotion and confidence scores for each class down below")

    col1, col2 = st.columns(2)
    # Define the labels and their positions

    with col1:
        st.subheader("Using own model")

        st.write('Dominant emotion: ', dominant_emotion)
        st.write(obj)

        st.subheader("Using Deepface library")

        st.write('Dominant emotion: ', analysis[0]['dominant_emotion'])
        st.write(analysis[0]['emotion'])

    with col2:
        # SVM prediction
        if svm_model_exists():
            im = cv2.imread(example_face_img_path)

            st.subheader("Using own SVM")
            st.write("Dominant emotion: ", predicted_class)

            st.write(svm_dictionary)

        st.subheader("Resmasknet classifier")

        st.write('Dominant emotion: ', results[0]['emo_label'])

        st.write(rmn_dict_proba)
    st.subheader("Credits to the following projects:")
    st.write('Deepface : https://github.com/serengil/deepface')
    st.write('ResmaskNet : https://github.com/phamquiluan/ResidualMaskingNetwork')

with tab2:
    group_image_box = st.checkbox("Use group images")
    # loaded_svm_model = setup_svm()
    if group_image_box:
        group_image_path = st.sidebar.selectbox("Group Images", group_images)
        if group_image_path != "NA":
            st.title("Here is the image you've selected")
            st.write(
                'Notice how the 2 face detection classifiers perform. On the right the neural network trained classifier is much better at finding faces')

            haar_column, dnn_column = st.columns(2)
            haar_column.subheader('HaarCascade classifier')

            marked_image, nr_faces, faces_frames = get_marked_image(group_image_path)
            haar_column.metric(label="Faces found", value=nr_faces)
            haar_column.image(marked_image)

            dnn_column.subheader("Classifier pre-trained on Res10")
            annotated_img, nr_faces2, faces_extracted = face_detect_NN(group_image_path, 0.7)
            dnn_column.metric(label="Faces found", value=nr_faces2)
            dnn_column.image(annotated_img)

            face_idx = 0
            btn = dnn_column.button("Calculate confidence scores", key=face_idx)
            for face in faces_extracted:
                gray_el = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                pil_image = Image.fromarray(gray_el)
                pil_image = pil_image.resize((48, 48))
                obj, dominant_emotion = get_prediction_deepface_way(face)
                dnn_column.image(face)

                if btn:
                    # expander section
                    expander2 = dnn_column.expander("See models probabilities")
                    expander2.subheader("Using own model")
                    expander2.write(dominant_emotion)
                    expander2.write(obj)

                    if face_idx <= nr_faces2:
                        face_path = faces_folder_path + '/face_' + str(face_idx) + '.jpg'

                        expander2.subheader('Using Deepface library')
                        face_analysis = DeepFace.analyze(face_path, actions=['emotion'], detector_backend='mtcnn',
                                                         enforce_detection=False)
                        face_analysis[0]['emotion'] = format_dictionary_probs(face_analysis[0]['emotion'])

                        expander2.write(face_analysis[0]['dominant_emotion'])
                        expander2.write(face_analysis[0]['emotion'])
                        predicted_class, probabilities = svm_get_predict(face_path, svm_model_aws)

                        expander2.subheader("Using own SVM")
                        svm_dictionary = get_dictonary_probs(probabilities)
                        expander2.write(predicted_class)
                        expander2.write(svm_dictionary)

                        expander2.subheader("Using ResMaskNet")

                        results = rmn.detect_emotion_for_single_frame(face)
                        rmn_dict_proba = get_dictionary_probs_RMN(results)

                        expander2.write(results[0]['emo_label'])
                        expander2.write(rmn_dict_proba)

                    face_idx += 1
        else:
            st.write("You can also select an image from left bar")

with tab3:
    single_images_box = st.checkbox('Use singular images')
    # loaded_svm_model = setup_svm()

    if single_images_box:
        selected_image_path = st.sidebar.selectbox("Singular subjects", all_images)
        if selected_image_path != "NA":

            haar_column, dnn_column = st.columns(2)

            haar_column.subheader('Face detectoru using HaarCascade classifier ')
            marked_image, nr_faces, faces_frames = get_marked_image(selected_image_path)
            haar_column.metric(label="Faces found", value=nr_faces)
            haar_column.image(marked_image)

            dnn_column.subheader("Face detector using pre-trained model on Res10")
            annotated_img, nr_faces2, faces_extracted = face_detect_NN(selected_image_path, 0.7)
            dnn_column.metric(label="Faces found", value=nr_faces2)
            dnn_column.image(annotated_img)

            for face in faces_extracted:
                # dnn_column.image(face)
                obj, dominant_emotion = get_prediction_deepface_way(face)
                dnn_column.subheader("Using own model")
                dnn_column.write(dominant_emotion)
                dnn_column.write(obj)

                dnn_column.subheader('Using Deepface library')
                face_analysis = DeepFace.analyze(faces_folder_path + '/face_0.jpg',
                                                 actions=['emotion'],
                                                 detector_backend='mtcnn',
                                                 enforce_detection=False)
                face_analysis[0]['emotion'] = format_dictionary_probs(face_analysis[0]['emotion'])

                dnn_column.write(face_analysis[0]['dominant_emotion'])
                dnn_column.write(face_analysis[0]['emotion'])

                predicted_class, probabilities = svm_get_predict(faces_folder_path + "/face_0.jpg", svm_model_aws)
                dnn_column.subheader("using own SVM")
                svm_dictionary = get_dictonary_probs(probabilities)
                dnn_column.write(predicted_class)
                dnn_column.write(svm_dictionary)
        else:
            st.write("You can also select an image from left bar")
