import altair as alt
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

@st.cache_data
def load_df():
    test_df = pd.read_csv("Dataframes/BIGdataset_face_test_df.csv")
    train_df = pd.read_csv("Dataframes/BIGdataset_face_train_df.csv")
    valid_df = pd.read_csv("Dataframes/BIGdataset_face_valid_df.csv")
    return test_df, train_df, valid_df


st.subheader("DATASET SUMMARY")
st.write("For this project I aim to use a combined dataset formed of 3 datasets (FER2013,CK+ and MMAFEDB) ."
         " These are publicly available and can be donwloaded from Kaggle.")

st.write("After a couple of iterations, i have compiled a subset of the original dataset with only pictures that contain frontal faces and can be detected. "
         "Here's how the datasets looks.")

st.write(
    "The Y-axis represents the nr of images and X-axis , the classes / categories of images. As you can see the dataset is imbalanced as "'neutral'" and "'happy'" have the most samples.")

test_df, train_df, valid_df = load_df()

col1, col2, col3 = st.columns(3)
colors = ["red", "green", "blue"]
labels = ['anger', 'fear', 'happy', 'neutral', 'sadness', 'surprise']


color_scale = alt.Scale(scheme='category10')

def get_bars_data(dataframe):
    values = []
    for l in labels:
        count = dataframe['class_names'].value_counts()[l]
        values.append(count)

    data = {'emotion_class': labels,
            'count': values}
    df = pd.DataFrame(data)

    bars = alt.Chart(df).mark_bar().encode(
        x='emotion_class',
        y='count',
        color=alt.Color('emotion_class', scale=color_scale)
    ).properties()

    return bars

with col1:
    st.header("Training")
    bars = get_bars_data(train_df)

    st.altair_chart(bars,use_container_width=True)

with col2:
    st.header("Validation")
    bars = get_bars_data(valid_df)
    st.altair_chart(bars,use_container_width=True)

with col3:
    st.header("Testing")
    bars = get_bars_data(test_df)
    st.altair_chart(bars,use_container_width=True)

st.subheader("FER2013 - DATASET SUMMARY")
fer_col1,fer_col2,fer_col3 = st.columns(3)


def fer_load_df():
    test_df = pd.read_csv("Dataframes/fer_testing.csv")
    train_df = pd.read_csv("Dataframes/fer_training.csv")
    valid_df = pd.read_csv("Dataframes/fer_valid.csv")
    return test_df, train_df, valid_df

test_df, train_df, valid_df = fer_load_df()

with fer_col1:
    st.header("Training")
    bars = get_bars_data(train_df)

    st.altair_chart(bars,use_container_width=True)

with fer_col2:
    st.header("Validation")
    bars = get_bars_data(valid_df)
    st.altair_chart(bars,use_container_width=True)

with fer_col3:
    st.header("Testing")
    bars = get_bars_data(test_df)
    st.altair_chart(bars,use_container_width=True)




