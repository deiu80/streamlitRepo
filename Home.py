import streamlit as st

st.set_page_config(
    page_title="CE301 - Project",
)
st.title("Welcome to my CE301 project")


st.subheader("INTRODUCTION")

st.write("Facial expressions play a crucial role in conveying emotions and "
         + "understanding human behavior. In recent years, deep learning has shown remarkable progress in the field of computer vision, especially in object recognition and image classification."
         + "In this project, I aim to leverage the power of convolutional neural networks (CNNs) to classify emotions from facial expressions.")

st.write(
    "My goal is to develop a model that can accurately identify different emotions such as happiness, sadness, anger, surprise, and disgust from facial images."
    " To achieve this, we will be using a large dataset of labeled facial expressions and develop a CNN model for this data. The final model will be able to recognize emotions from new facial expressions and provide a prediction for the emotion being displayed.")

st.write(
    "The findings from this application can have several practical applications in areas such as psychology, human-computer interaction, and security.")
st.subheader('Sections of the website')

link = '[Deep Learning in Computer vision](https://ce301project.streamlit.app/How_models_work)'


link2 = '[Webcam capture](https://ce301project.streamlit.app/Webcam_capture)'
link4= '[Emotion Detection and Analysis](https://ce301project.streamlit.app/Compare_Classifiers)'
link3 = '[Datasets](https://ce301project.streamlit.app/Datasets)'
st.markdown(link4, unsafe_allow_html=True)
st.markdown(link, unsafe_allow_html=True)
st.markdown(link2, unsafe_allow_html=True)
st.markdown(link3, unsafe_allow_html=True)

st.info("You can also select the pages from the left sidebar",icon="ℹ️")