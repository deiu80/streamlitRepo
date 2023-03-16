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