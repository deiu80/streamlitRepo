import streamlit as st
import cv2
from deepface import DeepFace

from helpers import single_face_img_path, setup_svm, svm_get_predict, svm_model_exists, format_dictionary_probs, \
    get_prediction_of_own_CNN, get_dictonary_probs_from_CNN

st.set_page_config(layout="wide")

def annotate_example_image(img, labels):
    label_positions = {
        "top_left": (10, 20),
        "top_right": (img.shape[1] - 5 - len(labels["top_right"]) * 11, 20),
        "bottom_left": (10, img.shape[0] - 10)
    }
    # Define the font and font scale to use for the labels
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6

    # Loop through each label and draw it on the image
    for label, text in labels.items():
        cv2.putText(img, text, label_positions[label], font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def get_face_coordinates(dict_of_detections):
    '''
    :param dict_of_detections: from DeepFace.extract_faces() :
    :return: coordinates for the first face region detected
    '''
    coordinates_obj = dict_of_detections[0]['facial_area']
    x = coordinates_obj['x']
    y = coordinates_obj['y']
    w = coordinates_obj['w']
    h = coordinates_obj['h']

    return x, y, w, h

labels = {
    "top_left": "CNN model: ",
    "top_right": "SVM model: ",
    "bottom_left": "Deepface: "
}

svm_model_aws = setup_svm()

st.title("Compare the classifiers ")
st.write("We do have 3 models available: custom CNN model, SVM and DeepFace ")


detections = DeepFace.extract_faces(img_path=single_face_img_path,
                                        detector_backend="ssd",
                                        enforce_detection=False)
x, y, width, height = get_face_coordinates(detections)

example_image = cv2.imread(single_face_img_path)
example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
face_detected = example_image[y:y + height, x:x + width]
cv2.rectangle(example_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
st.write("Here's the face we detected in the example image")

# OWN CNN
obj, dominant_emotion = get_prediction_of_own_CNN(face_detected)
labels['top_left'] += dominant_emotion + " " + str(round(obj[dominant_emotion], 2))

# Deepface
analysis = DeepFace.analyze(single_face_img_path, actions=['emotion'], detector_backend='mtcnn')
analysis[0]['emotion'] = format_dictionary_probs(analysis[0]['emotion'])
dominant_emotion_deepface = analysis[0]['dominant_emotion']
labels["bottom_left"] += analysis[0]['dominant_emotion'] + " " + str(
    round(analysis[0]['emotion'][dominant_emotion_deepface], 2))

if svm_model_exists():
    # SVM prediction
    predicted_class, probabilities = svm_get_predict(svm_model_aws, face_img=face_detected)
    svm_dictionary = get_dictonary_probs_from_CNN(probabilities)
    labels['top_right'] += predicted_class + ' ' + str(round(svm_dictionary[predicted_class], 2))

annoted_image = annotate_example_image(example_image, labels)
st.image(annoted_image)
st.write("You can notice the predicted enotion and confidence scores for each class down below")

col1, col2, col3 = st.columns(3)
# Define the labels and their positions

with col1:
    st.subheader("Using own model")

    st.write('Dominant emotion: ', dominant_emotion)
    st.write(obj)
if svm_model_exists():
    with col3:
        # SVM prediction
        st.subheader("Using SVM")
        st.write("Dominant emotion: ", predicted_class)
        st.write(svm_dictionary)

with col2:
    st.subheader("Using Deepface")
    st.write('Dominant emotion: ', analysis[0]['dominant_emotion'])
    st.write(analysis[0]['emotion'])

st.info("Credits to the following projects:")
st.write('Deepface : https://github.com/serengil/deepface')