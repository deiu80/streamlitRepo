import pathlib
import pickle
import cv2
import streamlit as st
import keras_preprocessing
import tensorflow
from keras_preprocessing.image import load_img
import numpy as np
from keras.models import load_model
from skimage.feature import hog
import boto3, os, time


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    page_title_str = "CE301 - Image uploader"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


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


images_folder_path = "./Images"
faces_folder_path = "./Faces"

extracted_face_path = images_folder_path + "/extracted_face.jpg"
enhanced_face_path = images_folder_path + "/enhanced_face.jpg"

labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']


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
def get_classifier():
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    default_classifier = cv2.CascadeClassifier(str(cascade_path))
    return default_classifier


def get_extracted_face_path():
    return extracted_face_path


def get_enhanced_face_path():
    return enhanced_face_path


def prepare_image_for_svm(img):
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


def image_resizer(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


@st.cache_data
def get_prediction_deepface_way(image, img_path='default'):
    obj = {}
    obj["emotion"] = {}
    #  deepface way
    if img_path != 'default':
        img = cv2.imread(img_path)
    else:
        img = image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (48, 48))
    img_gray = keras_preprocessing.image.img_to_array(img_gray)
    img_gray = np.expand_dims(img_gray, axis=0)
    img_gray /= 255
    loaded_model = load_model('./best_model_optimised_cnn.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    deepface_emotion_predictions = loaded_model.predict(img_gray, verbose=1)[0, :]
    sum_of_predictions = deepface_emotion_predictions.sum()

    for i, emotion_label in enumerate(labels):
        print(deepface_emotion_predictions[i])
        emotion_prediction = round(100 * deepface_emotion_predictions[i] / sum_of_predictions, 5)
        obj["emotion"][emotion_label] = emotion_prediction
    obj["dominant_emotion"] = labels[np.argmax(deepface_emotion_predictions)]
    print('deepface', deepface_emotion_predictions)

    return obj["emotion"], obj["dominant_emotion"]


@st.cache_data()
def get_model_prediction(_image):
    '''
    :param image of 48*48 size and grayscale
    :return: name of the predicted class, e.g happy, sad
    '''
    # old way
    img_array = keras_preprocessing.image.img_to_array(_image)
    img_array = np.expand_dims(img_array, axis=0)
    image_input = np.vstack([img_array])
    loaded_model = load_model('./best_model_optimised_cnn.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #  make predictions of the model
    prediction = loaded_model.predict(image_input)

    # Get the confidence score for each class
    class_probabilities = tensorflow.nn.softmax(prediction[0]).numpy()

    confidences = []
    for lab, prob in zip(labels, class_probabilities):
        confidences.append(round(prob * 100, 5))

    return labels[np.argmax(prediction)], confidences


# above method is for reading images from disk
def get_img_face_frame(img_path):
    nr_of_faces = 0
    cv2_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    faces = get_viola_classifier().detectMultiScale(gray_img,
                                                    scaleFactor=1.2,
                                                    minNeighbors=5,
                                                    minSize=(30, 30),
                                                    flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)

    for (x, y, width, height) in faces:
        cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 4)
        face_frame = gray_img[y:height + y, x:x + width]
        nr_of_faces += 1
    if faces != ():
        cv2.imwrite(images_folder_path + "/extracted_face.jpg", face_frame)
    st.write("Nr of faces found: " + str(nr_of_faces))
    return cv2.imread(images_folder_path + "/extracted_face.jpg")


def get_all_faces(img_path):
    list_of_faces = []
    cv2_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt.xml"
    clf = cv2.CascadeClassifier(str(cascade_path))

    faces = clf.detectMultiScale(gray_img,
                                 scaleFactor=1.2,
                                 minNeighbors=3,
                                 minSize=(50, 50),
                                 flags=cv2.CASCADE_DO_CANNY_PRUNING)

    for (x, y, width, height) in faces:
        face_frame = gray_img[y:height + y, x:x + width]
        list_of_faces.append(face_frame)
    print("nr of faces found=", len(list_of_faces))

    return list_of_faces


@st.cache_data()
def get_marked_image(img_path):
    modified_img = cv2.imread(img_path)
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


@st.cache_data
def face_detect_NN(image_path, threshold):
    original_image = cv2.imread(image_path)
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    frameHeight = original_image.shape[0]
    frameWidth = original_image.shape[1]

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(original_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Set the input to the model
    net.setInput(blob)
    bboxes = []
    # Forward pass through the network
    detections = net.forward()

    # Loop over the detections
    for i in range(10):
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

    # writing the faces extracted to the folder
    for f, nr in zip(faces_frames, range(len(face_frame))):
        cv2.imwrite(faces_folder_path + '/face_' + str(nr) + '.jpg', cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), len(bboxes), faces_frames


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


@st.cache_resource
def svm_get_predict(img_path, _loaded_model):
    features = []
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = image_resizer(im, 64, 64)
    fd1, hog_image = hog(im, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm='L2-Hys',
                         transform_sqrt=False, visualize=True)
    cv2.imwrite(images_folder_path + '/hog_image.jpg', hog_image)
    features.append(fd1)
    # extract features from the image
    features = np.array(features)

    # use the SVM model to make a prediction
    predicted_class = _loaded_model.predict(features)
    probabilities = _loaded_model.predict_proba(features)

    return predicted_class[0], probabilities[0]
