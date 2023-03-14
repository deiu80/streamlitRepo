# import codecs
# import pathlib
#
# import av
# import cv2
# import numpy as np
import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# from helpers import get_model_prediction, image_resizer, image_enhancer
# import streamlit.components.v1 as stc

st.set_page_config(layout="wide")

st.title("Example using JavaScript")

st.subheader('Head over to:')
st.write('https://deiu80.github.io/live-webcam')

import streamlit as st

video_file = open('webcam_video.mov', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
#
#
# def html_component(calc_html, width=1000, height=1200):
#     calc_file = codecs.open(calc_html, 'r')
#     page = calc_file.read()
#
#     stc.html(page, width=width, height=height, scrolling=False)
#
#
# html_component(calc_html='html/index.html')

#
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
#
# clf = cv2.CascadeClassifier(str(cascade_path))
#
# st.title("Press Start to record the webcam")
#
#
# def process_img(image):
#     color = (0, 0, 255)
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = clf.detectMultiScale(gray_img,
#                                  scaleFactor=1.1,
#                                  minNeighbors=5,
#                                  minSize=(30, 30),
#                                  flags=cv2.CASCADE_SCALE_IMAGE)
#     for (x, y, width, height) in faces:
#         rectangle_img = cv2.rectangle(gray_img, (x, y), (x + width, y + height), (0, 255, 0), 2),
#         face_frame_image = gray_img[y:height + y, x:x + width]
#
#     img = cv2.cvtColor(rectangle_img, cv2.COLOR_BGR2RGB)
#     return img
#
#
# def callback(frame):
#     rgb_frame = frame.to_rgb()
#     img = rgb_frame.to_ndarray(format="rgb24")
#     gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     faces = clf.detectMultiScale(gray_img,
#                                  scaleFactor=1.1,
#                                  minNeighbors=5,
#                                  minSize=(30, 30),
#                                  flags=cv2.CASCADE_SCALE_IMAGE)
#
#     for (x, y, width, height) in faces:
#         # Specify the color of the text
#         color = (0, 255, 0)
#         # Specify the thickness of the text
#         thickness = 1
#         # Specify the type of the line used to draw the text
#         line_type = cv2.LINE_AA
#         face_frame = gray_img[y:height + y, x:x + width]
#         face_frame = image_resizer(face_frame, 48,48)
#         face_frame = image_enhancer(face_frame)
#         # Add the text to the image
#         model_class_prediction, confidences = get_model_prediction(face_frame)
#         cv2.putText(img, model_class_prediction, (x + 67, y - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, line_type)
#         cv2.putText(img, confidences, (x + 17, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, line_type)
#         rectangle_img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 50), 2),
#         if rectangle_img:
#             a = np.array(rectangle_img)
#             return av.VideoFrame.from_ndarray(np.squeeze(a), format='rgb24')
#     return frame
#
#
# webrtc_streamer(key="example",
#                 video_frame_callback=callback,
#                 rtc_configuration={
#                     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#                 },
#                 media_stream_constraints={"audio": False,
#                                           "video": {
#                                               "width": {"min": 640, "ideal": 1200, "max": 1200}}}
#                 )
