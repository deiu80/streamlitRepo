import streamlit as st
from keras.models import load_model


def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")


def page2():
    st.markdown("# Real Time️")
    st.sidebar.markdown("# Real time feed")


loaded_model = load_model('./best_model_optimised_cnn.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
