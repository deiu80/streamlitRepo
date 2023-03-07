import os

import cv2
import streamlit as st
from PIL import Image
from deepface import DeepFace
from rmn import RMN


from helpers import face_detect_NN, get_marked_image, labels, faces_folder_path, get_prediction_deepface_way, \
    svm_get_predict, setup_svm, prepare_image_for_svm
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

single_face_img_path = 'Images/single_face.jpg'

svm_det = None
resmask_det = None

svm_model_aws = setup_svm()

# @st.cache_data(show_spinner=True)
# def load_detectors():
#     global svm_det, resmask_det
#     svm_det = Detector(emotion_model='svm')
#     resmask_det = Detector(emotion_model='resmasknet')
#
#
# load_detectors()

tab1, tab2, tab3 = st.tabs(["Example", "Group Images", "Single Images"])


@st.cache_resource
def predictions(_svm_det, _resmas_det, img_path):
    face_prediction_using_resk = _resmas_det.detect_image(img_path)
    face_prediction_using_svm = _svm_det.detect_image(img_path)
    return face_prediction_using_svm, face_prediction_using_resk


@st.cache_data
def face_predictions(_svm_det, _resmas_det, img_path):
    face_prediction_using_resk = _resmas_det.detect_image(img_path)
    face_prediction_using_svm = _svm_det.detect_image(img_path)
    return face_prediction_using_svm, face_prediction_using_resk


def format_dictionary_probs(analysis):
    for k in analysis:
        analysis[k] = round(analysis[k], 5)
    return analysis


@st.cache_data
def get_dictonary_probs(probabilities):
    obj = {}
    for i, emotion_label in enumerate(labels):
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


with tab1:

    st.title("Compare the classifiers ")
    st.write("We do have 4 models available: own model,own SVM classifier, DeepFace and ResMasknet ")
    st.write('Deepface Github https://github.com/serengil/deepface')
    st.write('ResmaskNet https://github.com/phamquiluan/ResidualMaskingNetwork')

    returned_img, nr, faces_extracted = face_detect_NN(single_face_img_path, 0.7)
    analysis = DeepFace.analyze(single_face_img_path, actions=['emotion'], detector_backend='mtcnn')
    st.image(returned_img)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Class probabilities ")
        st.subheader("Using own model")
        obj, dominant_emotion = get_prediction_deepface_way(returned_img, 'Faces/face_0.jpg')
        st.write('Dominant emotion: ',dominant_emotion)
        st.write(obj)

        st.subheader("Using Deepface library")
        analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])
        st.write('Dominant emotion: ', analysis[0]['dominant_emotion'])
        st.write(analysis[0]['emotion'])

    with col2:
        # SVM prediction
        im = cv2.imread('Faces/face_0.jpg')

        predicted_class, probabilities = svm_get_predict("Faces/face_0.jpg", svm_model_aws)
        st.subheader("Using own SVM")
        st.write("Dominant emotion: ", predicted_class)

        dictionary = get_dictonary_probs(probabilities)
        st.write(dictionary)

        # RMN prediction
        rmn = load_RMN()
        st.subheader("Resmasknet classifier")
        im = cv2.imread(single_face_img_path)
        results = rmn.detect_emotion_for_single_frame(im)
        st.write('Dominant emotion: ', results[0]['emo_label'])
        dict_proba = get_dictionary_probs_RMN(results)
        st.write(dict_proba)

with tab2:
    group_image_box = st.checkbox("Use group images")
    # loaded_svm_model = setup_svm()
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

            dnn_column.subheader("Classifier pre-trained on Res10 ")
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
                        dictionary = get_dictonary_probs(probabilities)
                        expander2.write(predicted_class)
                        expander2.write(dictionary)
                        # svm_detections, resmask_detections = face_predictions(svm_det, resmask_det, face_path)
                        # expander2.subheader("Using SVM classifier")
                        # expander2.write(svm_detections.emotions)
                        #
                        # expander2.subheader("Using resmasknet classifier")
                        # expander2.write(resmask_detections.emotions)
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
                face_analysis = DeepFace.analyze(faces_folder_path + '/face_0.jpg', actions=['emotion'],
                                                 detector_backend='mtcnn',
                                                 enforce_detection=False)
                face_analysis[0]['emotion'] = format_dictionary_probs(face_analysis[0]['emotion'])

                dnn_column.write(face_analysis[0]['dominant_emotion'])
                dnn_column.write(face_analysis[0]['emotion'])

                predicted_class, probabilities = svm_get_predict(faces_folder_path + "/face_0.jpg", svm_model_aws)
                dnn_column.subheader("using own SVM")
                dictionary = get_dictonary_probs(probabilities)
                dnn_column.write(predicted_class)
                dnn_column.write(dictionary)

                # svm_detections, resmask_detections = predictions(svm_det, resmask_det, single_face_img_path)
                # dnn_column.subheader("Using SVM classifier")
                # dnn_column.write(svm_detections.emotions)
                #
                # dnn_column.subheader("Using resmasknet classifier")
                # dnn_column.write(resmask_detections.emotions)
        else:
            st.write("You can also select an image from left bar")
