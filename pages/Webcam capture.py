import cv2
from deepface import DeepFace

from helpers import faces_folder_path, svm_model_exists, svm_get_predict, get_prediction_of_own_CNN, face_detect_NN, \
    loading_RMN, face_detect_RMN
from helpers import format_dictionary_probs, get_dictonary_probs, get_dictionary_probs_RMN,setup_svm

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="CE301 - Capture from Webcam Live Feed",
    layout="wide"
)

st.title("Capture from Webcam Live Feed")

img_file_buffer = st.camera_input("Take a frontal picture")

capture_path = faces_folder_path + "/rgb_capture.png"
capture_face_path = faces_folder_path + "/capture_face.jpg"

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    rgb_capture = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(capture_path, rgb_capture)

    # returned_img, nr, faces_extracted_list = face_detect_NN('', rgb_capture)
    rmn = loading_RMN()
    returned_img, nr, faces_extracted_list = face_detect_RMN(rgb_capture, _rmn=rmn)


    # new way
    if nr == 0:
        st.warning("Couldn't find a face. Try again!")
    else:
        st.write("Here's the faces we found")
        st.image(returned_img)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Using own model")
            # OWN CNN
            obj, dominant_emotion = get_prediction_of_own_CNN(rgb_capture)

            st.write('Dominant emotion: ', dominant_emotion)
            st.write(obj)
            # Deepface
            analysis = DeepFace.analyze(capture_path, actions=['emotion'], detector_backend='mtcnn')
            analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])

            st.subheader("Using Deepface library")

            st.write('Dominant emotion: ', analysis[0]['dominant_emotion'])
            st.write(analysis[0]['emotion'])

        with col2:
            # SVM prediction
            if svm_model_exists():
                svm_model_aws = setup_svm()
                # SVM prediction
                if nr == 1:
                    detected_face = faces_extracted_list[0]

                    predicted_class, probabilities = svm_get_predict(capture_face_path, svm_model_aws, detected_face)
                    svm_dictionary = get_dictonary_probs(probabilities)

                    st.subheader("Using own SVM")
                    st.write("Dominant emotion: ", predicted_class)

                    st.write(svm_dictionary)


            results = rmn.detect_emotion_for_single_frame(rgb_capture)
            rmn_dict_proba = get_dictionary_probs_RMN(results)
            st.subheader("Resmasknet classifier")

            st.write('Dominant emotion: ', results[0]['emo_label'])

            st.write(rmn_dict_proba)
