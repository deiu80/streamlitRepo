import pathlib
import cv2
import streamlit as st
import keras_preprocessing
from keras_preprocessing.image import load_img
import numpy as np
from keras.models import load_model


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


images_folder_path = "./Images"

extracted_face_path = images_folder_path + "/extracted_face.jpg"
enhanced_face_path = images_folder_path + "/enhanced_face.jpg"

labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

viola_jones = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt.xml"
viola_jones_classifier = cv2.CascadeClassifier(str(viola_jones))

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
default_classifier = cv2.CascadeClassifier(str(cascade_path))


def get_viola_classifier():
    return viola_jones_classifier


def get_classifier():
    return default_classifier


def get_extracted_face_path():
    return extracted_face_path


def get_enhanced_face_path():
    return enhanced_face_path


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


def image_resizer(image, target_size=48):
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)


def get_model_prediction(image):
    '''
    :param image of 48*48 size and grayscale
    :return: name of the predicted class, e.g happy, sad
    '''
    img_array = keras_preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    image_input = np.vstack([img_array])
    loaded_model = load_model('./best_model_optimised_cnn.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #  make predictions of the model
    prediction = loaded_model.predict(image_input)

    return labels[np.argmax(prediction)]


# above method is for reading images from disk
def get_img_face_frame(img_path):
    nr_of_faces = 0
    cv2_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    faces = default_classifier.detectMultiScale(gray_img,
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

    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), len(bboxes), faces_frames
