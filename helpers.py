import pathlib
import pickle
import cv2
import numpy
import streamlit as st
import keras_preprocessing
from keras_preprocessing.image import load_img
import numpy as np
from keras.models import load_model

from skimage.feature import hog
import boto3, os, time

images_folder_path = "./Images"
faces_folder_path = "./Faces"
single_face_img_path = 'Images/single_face.jpg'

example_face_img_path = 'Faces/example_image.jpg'

extracted_face_path = images_folder_path + "/extracted_face.jpg"
enhanced_face_path = images_folder_path + "/enhanced_face.jpg"

emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']


def download_model_from_aws(file_name_in_aws):
    """Accessing the S3 buckets using boto3 client"""
    # file_name_in_aws = 'svm_model_only_faces.sav'
    s3_bucket_name = 'for-streamlit'
    s3 = boto3.client('s3',
                      aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'])
    print("client_ready")
    s3.download_file(s3_bucket_name, file_name_in_aws, file_name_in_aws)
    print("downloaded_file:", file_name_in_aws)


def svm_model_exists():
    filename = "svm_model_only_faces.sav"
    if os.path.exists(filename):
        return True
    return False


@st.cache_resource
def get_viola_classifier():
    viola_jones = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt.xml"
    viola_jones_classifier = cv2.CascadeClassifier(str(viola_jones))
    return viola_jones_classifier


def get_extracted_face_path():
    return extracted_face_path


def preprocess_image_for_svm(img):
    # convert it to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(img)

    # Apply image denoising
    denoised_image = cv2.fastNlMeansDenoising(equalized_image, None, 10, 7, 21)
    # apply Laplacian sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filtered = cv2.filter2D(denoised_image, -1, kernel)

    return filtered


@st.cache_data(show_spinner=False)
def image_resizer(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


@st.cache_data(show_spinner=False)
def face_detect_RMN(img_path, _rmn):
    if type(img_path) == numpy.ndarray:
        image = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    list_detections = _rmn.detect_faces(image)
    face_frames = []
    for box in list_detections:
        face_frame = image[box['ymin'] + 5:box["ymax"] - 5, box["xmin"] + 5:box["xmax"] - 5]
        cv2.rectangle(image, (box["xmin"], box['ymin']),
                      (box["xmax"], box["ymax"]),
                      (0, 255, 0), 2
                      )

        face_frames.append(face_frame)

    return image, len(list_detections), face_frames


@st.cache_data(show_spinner=False)
def get_prediction_of_own_CNN(image, img_path='default'):
    obj = {"emotion": {}}
    #  deepface way
    if img_path != 'default':
        img = cv2.imread(img_path)
    else:
        img = image

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = image_resizer(img_gray, 48, 48)
    img_gray = keras_preprocessing.image.img_to_array(img_gray)
    img_gray = np.expand_dims(img_gray, axis=0)
    img_gray /= 255
    loaded_model = load_model('./best_model_optimised_cnn.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    emotion_predictions = loaded_model.predict(img_gray, verbose=1)[0, :]

    for i, emotion_label in enumerate(emotion_labels):
        emotion_prediction = round(100 * emotion_predictions[i], 3)
        obj["emotion"][emotion_label] = emotion_prediction

    obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

    return obj["emotion"], obj["dominant_emotion"]


def get_marked_image(img_path, _img=None):
    if _img is None:
        modified_img = cv2.imread(img_path)
    else:
        modified_img = _img
    gray_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)

    faces = get_viola_classifier().detectMultiScale(gray_img,
                                                    scaleFactor=1.1,
                                                    minNeighbors=3,
                                                    minSize=(30, 30),
                                                    flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
    faces_frames = []
    for (x, y, width, height) in faces:
        face_frame = modified_img[y:y + height, x:x + width]
        faces_frames.append(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
        cv2.rectangle(modified_img, (x, y), (x + width, y + height), (0, 212, 90), 3)
    return cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB), len(faces), faces_frames


def setup_svm():
    filename = "svm_model_only_faces.sav"
    if os.path.isfile(filename):
        # model exists
        loaded_svm_model = pickle.load(open(filename, 'rb'))
        return loaded_svm_model
    else:
        st.warning('Svm model does not exist. Click the button below.')
        download_button = st.button(label="Download SVM model from AWS")

        if download_button:
            download_model_from_aws(filename)
            print("Starting...")
            time.sleep(6)
            loaded_svm_model = pickle.load(open(filename, 'rb'))

            return loaded_svm_model
        return None


@st.cache_data
def svm_get_predict(_loaded_model, face_capture_path=None, face_img=None):
    '''

    :param face_img: facial image
    :param face_capture_path: This is the path of the image containing only the face (face_capture)
    :param _loaded_model: this is the SVM model that's loaded from local storage
    :return: predicted_class/emotion and a list of all emotions confidences
    '''

    features = []
    if face_img is None:
        im = cv2.imread(face_capture_path)
    else:
        im = face_img

    processed_face_im = preprocess_image_for_svm(im)

    processed_face_im = image_resizer(processed_face_im, 64, 64)
    fd1, hog_image = hog(processed_face_im, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
                         block_norm='L2-Hys',
                         transform_sqrt=False, visualize=True)
    features.append(fd1)
    # extract features from the image
    features = np.array(features)

    # use the SVM model to make a prediction

    probabilities = _loaded_model.predict_proba(features)
    index_of_max = np.array(probabilities[0]).argmax()
    predicted_class = emotion_labels[index_of_max]
    return predicted_class, probabilities[0]


def format_dictionary_probs(analysis):
    for k in analysis:
        analysis[k] = round(analysis[k], 3)
    return analysis


def get_dictonary_probs_from_CNN(probabilities):
    obj = {}
    for i, emotion_label in enumerate(emotion_labels):
        obj[emotion_label] = round(100 * probabilities[i], 3)
    return obj
