import streamlit as st
import pandas as pd
import numpy as np

st.subheader("DATASET SUMMARY")
st.write("For this project I aim to use a combined dataset formed of 3 datasets (FER2013,CK+ and MMAFEDB) ."
         " These are publicly available and can be donwloaded from Kaggle.")
st.write("Here's how the dataset looks.")

st.image("Plots/dataset_summary.png")
st.write("The Y-axis represents the nr of images and X-axis , the classes / categories of images. As you can see the dataset is imbalanced as "'neutral'" and "'happy'" have the most samples.")
st.subheader("FER2013 - DATASET SUMMARY")
st.image("Plots/fer2013_summary.png")

# TODO - get pandas dataframes imported and displayed with bar charts instead of pictures
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=["a", "b" ])
#
#
#
# col1, col2, col3 = st.columns(3)
#
# with col1:
#    st.header("A cat")
#    st.bar_chart(chart_data)
#
# with col2:
#    st.header("A dog")
#    st.bar_chart(chart_data)
#
# with col3:
#    st.header("An owl")
#    st.bar_chart(chart_data)