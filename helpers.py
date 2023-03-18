import pathlib
import pickle
import cv2
import numpy
import streamlit as st
import keras_preprocessing
import tensorflow
from keras_preprocessing.image import load_img
import numpy as np
from keras.models import load_model
from rmn import RMN

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


@st.cache_resource(show_spinner=False)
def loading_RMN():
    rmn = RMN()
    return rmn


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


@st.cache_resource
def get_default_classifier():
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    default_classifier = cv2.CascadeClassifier(str(cascade_path))
    return default_classifier


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
    # cv2.imwrite(images_folder_path + '/filtered_img.jpg', filtered)
    # print("filtered image written")
    return filtered


def image_enhancer(image):
    '''

    :param image:
    :return: transformed image
    '''

    # read the image
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # # apply Laplacian sharpening
    # kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # filtered = cv2.filter2D(equalized_image, -1, kernel)
    #
    # sharpened_image = equalized_image + filtered
    # # Apply image denoising
    # denoised_image = cv2.fastNlMeansDenoising(sharpened_image, None, 2, 7, 5)

    return equalized_image


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
def face_detect_NN(image_path, image=None):
    '''
    if you provide image_path and image variable, the latter will be used to avoid reading it from local
    :param image_path:
    :param threshold:
    :param image: optional parameter for an image variable
    :return:
    '''
    if image is not None:
        original_image = image
    else:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    threshold = 0.7
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    frameHeight = original_image.shape[0]
    frameWidth = original_image.shape[1]

    # Create a blob from the input image

    blob = cv2.dnn.blobFromImage(original_image, 1.0, (300, 300), (104.0, 177.0, 123.0),False,False)
    # Set the input to the model
    net.setInput(blob)
    bboxes = []
    # Forward pass through the network
    detections = net.forward()

    # Loop over the detections
    for i in range(len(detections)):
        # Get the confidence score of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])

    faces_frames = []

    for (x1, y1, x2, y2) in bboxes:
        face_frame = original_image[y1:y2, x1:x2]
        faces_frames.append(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    #  NO FACES FOUND case, we return the original image ,0 and the empty list
    if len(faces_frames) == 0:
        return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 0, faces_frames

        # writing the faces extracted to the folder
    if image_path == single_face_img_path:
        for f, nr in zip(faces_frames, range(len(face_frame))):
            cv2.imwrite(faces_folder_path + '/example_image.jpg', cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    else:
        for f, nr in zip(faces_frames, range(len(face_frame))):
            cv2.imwrite(faces_folder_path + '/face_' + str(nr) + '.jpg', cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    return original_image, len(bboxes), faces_frames


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
        emotion_prediction = round(100 * emotion_predictions[i], 5)
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
        cv2.rectangle(modified_img, (x, y), (x + width, y + height), (0, 255, 0), 5)
    return cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB), len(faces), faces_frames


def setup_svm():
    filename = "svm_model_only_faces.sav"
    if os.path.isfile(filename):
        # model exists
        loaded_svm_model = pickle.load(open(filename, 'rb'))
        return loaded_svm_model
    else:
        st.warning(filename + ' svm model does not exist')
        download_button = st.button(label="Download SVM model from AWS")

        if download_button:
            download_model_from_aws(filename)
            print("Starting...")
            time.sleep(6)
            loaded_svm_model = pickle.load(open(filename, 'rb'))

            return loaded_svm_model
        return None


def svm_get_predict(face_capture_path, _loaded_model, face_img=None):
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

    cv2.imwrite(images_folder_path + '/hog_image.jpg',   image_resizer(hog_image,224,224))
    features.append(fd1)
    # extract features from the image
    features = np.array(features)

    # use the SVM model to make a prediction
    predicted_class = _loaded_model.predict(features)
    probabilities = _loaded_model.predict_proba(features)

    return predicted_class[0], probabilities[0]


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
