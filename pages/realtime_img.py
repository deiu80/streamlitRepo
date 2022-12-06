import pathlib
from os.path import exists

import streamlit as st
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

btn = st.button("Capture face")

camera = cv2.VideoCapture(0)
path = "./Images"
while run:
    _, frame = camera.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if btn:
        faces = clf.detectMultiScale(gray_img,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, width, height) in faces:
            cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            face_frame = gray_img[y:height + y, x:x + width]
            cv2.imwrite(path+"/extracted_face.jpg", face_frame)
        st.write("Extracted face is:")
        if exists(path+"/extracted_face.jpg"):
            st.image(cv2.imread(path+"/extracted_face.jpg"))
        camera.release()
        break
else:

    st.write("Stopped")
    if exists(path + "/extracted_face.jpg"):
        st.write("Extracted face is:")
        st.image(cv2.imread(path+"/extracted_face.jpg"))



