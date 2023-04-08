import cv2
from deepface import DeepFace
from helpers import faces_folder_path, svm_model_exists, svm_get_predict, get_prediction_of_own_CNN

from helpers import format_dictionary_probs, get_dictonary_probs_from_CNN, setup_svm

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
    list_of_dicts = DeepFace.extract_faces(img_path=rgb_capture,
                                           target_size=(150,150),
                                           detector_backend="dlib",
                                           enforce_detection=False)
    # print(list_of_dicts)
    confidence = list_of_dicts[0]['confidence']


    # rmn = loading_RMN()
    # returned_img, nr, faces_extracted_list = face_detect_RMN(rgb_capture, _rmn=rmn)
    if confidence <= 0.5:
        st.warning("Couldn't find a face. Try again!")
    else:
        st.write("Here's the faces we found")
        for dic in list_of_dicts:

            facial_area_coordinates = dic['facial_area']
            x = facial_area_coordinates['x']
            y = facial_area_coordinates['y']
            width = facial_area_coordinates['w']
            height = facial_area_coordinates['h']
            rgb_capture = cv2.cvtColor(rgb_capture, cv2.COLOR_BGR2RGB)
            face_frame = rgb_capture[y:y + height, x:x + width]
            st.image(face_frame)


            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Using own model")
                # OWN CNN
                obj, dominant_emotion = get_prediction_of_own_CNN(face_frame)

                st.write('Dominant emotion: ', dominant_emotion)
                st.write(obj)

            with col2:
                # Deepface
                analysis = DeepFace.analyze(capture_path, actions=['emotion'], detector_backend='mtcnn')
                analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])

                st.subheader("Using Deepface")

                st.write('Dominant emotion: ', analysis[0]['dominant_emotion'])
                st.write(analysis[0]['emotion'])

            if svm_model_exists():
                with col3:
                   # SVM prediction
                    svm_model_aws = setup_svm()
                    # SVM prediction
                    if confidence > 0.5:
                        predicted_class, probabilities = svm_get_predict( svm_model_aws, face_img=face_frame)
                        svm_dictionary = get_dictonary_probs_from_CNN(probabilities)

                        st.subheader("Using own SVM")
                        st.write("Dominant emotion: ", predicted_class)

                        st.write(svm_dictionary)
